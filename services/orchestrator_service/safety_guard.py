"""
services/orchestrator_service/safety_guard.py
Deterministischer Pre-Retrieval Safety- & Krisen-Interceptor gemäß FA-06 (Human-Handoff)
und SA-06 (Diskretion & Schutz bei sensiblen Themen).

Arbeitet nach dem 3-Stufen-Triage-Modell:
1. ACUTE_CRISIS (Suizidalität, akute/sexualisierte Gewalt) -> Hard Interrupt + Verifizierte Notfallkontakte (Kein LLM!)
2. ADDICTION_CRISIS (Drogen/Sucht-Notlagen) -> Fester Sicherheitsnetz-Verweis auf DROBS Hildesheim
3. INAPPROPRIATE_NSFW (Explizit sexualisierte/missbräuchliche Eingaben am Terminal) -> Sachlicher Gesprächsabbruch
"""

import re
from typing import Dict, Optional, Tuple
from shared.schemas import AgentTraceStep, ChatRequest, ChatResponse, RobotState


# =====================================================================
# 1. TRIGGER-MUSTER (Mehrsprachig: DE, EN, TR, AR, UK, RU)
# =====================================================================

SUICIDE_TRIGGERS = [
    r"\bsuizid", r"\bselbstmord", r"\bbringen?\s+mich\s+um\b", r"\bwill\s+nicht\s+mehr\s+leben\b",
    r"\bleben\s+beenden\b", r"\bmich\s+umbringen\b", r"\bselbstverletz", r"\britzen\b",
    r"\bsterben\s+wollen\b", r"\bkeinen\s+ausweg\s+mehr\b",
    r"\bsuicide\b", r"\bkill\s+myself\b", r"\bend\s+my\s+life\b",
    r"\bintihar\b", r"\bkendimi\s+öldür",
    r"انتحار", r"أقتل نفسي",
    r"самогубств", r"суїцид", r"суицид", r"убить\s+себя"
]

VIOLENCE_ABUSE_TRIGGERS = [
    r"\bvergewaltig", r"\bsexuell(?:e|er|en)?\s+(?:missbrauch|übergriff|belästig|gewalt)",
    r"\bhäusliche\s+gewalt\b", r"\bschlägt\s+mich\b", r"\bmisshandel", r"\bfrauenhaus\b",
    r"\bbedroht\s+mich\b", r"\bzwangsverheirat", r"\bstalking\b",
    r"\brape\b", r"\bdomestic\s+violence\b", r"\bsexual\s+assault\b",
    r"\btecavüz\b", r"\bşiddet\b",
    r"اغتصاب", r"عنف منزلي",
    r"насильств", r"насили"
]

DRUG_EMERGENCY_TRIGGERS = [
    r"\büberdosis\b", r"\bdrogennotfall\b", r"\bentzugserscheinung", r"\bheroin", r"\bkokain",
    r"\bcrack\b", r"\bfentanyl\b", r"\bdrogensucht\b", r"\bdrogenabhängig", r"\bsuchtberatung\b",
    r"\bdrogenberatung\b", r"\balkoholsucht\b", r"\bspielsucht\b"
]

NSFW_TROLLING_TRIGGERS = [
    r"\bporno\b", r"\bpornografie\b", r"\bsexchat\b", r"\bnacktbild", r"\berotik\b",
    r"\bfick(?:en|dich)\b", r"\bhurensohn\b", r"\bschlampe\b"
]


# =====================================================================
# 2. VERIFIZIERTE NOTFALL-KARTEN & VCARD-QR-CODES (HILDESHEIM & BUND)
# =====================================================================

CRISIS_RESPONSE_SUICIDE: Dict[str, str] = {
    "de": (
        "Es tut mir sehr leid, dass Sie sich in einer so schweren Situation befinden. "
        "Sie sind damit nicht allein – bitte wenden Sie sich sofort an Menschen, die Ihnen jetzt professionell und kostenfrei beistehen:\n\n"
        "• **TelefonSeelsorge (24 Stunden, anonym & kostenlos):** `0800 111 0 111` oder `0800 111 0 222`\n"
        "• **Sozialpsychiatrischer Dienst Landkreis/Stadt Hildesheim:** Bischof-Janssen-Str. 31, 31134 Hildesheim | Tel.: `05121 309-7371`\n"
        "• **Akuter medizinischer Notruf (24/7):** `112`\n\n"
        "*(Zu Ihrem Schutz am Terminal wird diese Nachricht nicht laut vorgelesen. Sie können die Nummern über den QR-Code direkt auf Ihr Smartphone scannen.)*"
    ),
    "en": (
        "I am very sorry that you are going through such a difficult time. Please reach out immediately to free, confidential professional support:\n\n"
        "• **Crisis Hotline Germany (TelefonSeelsorge, 24/7, free):** `0800 111 0 111`\n"
        "• **Social Psychiatric Service Hildesheim:** Bischof-Janssen-Str. 31, 31134 Hildesheim | Phone: `05121 309-7371`\n"
        "• **Emergency Services (24/7):** `112`"
    ),
    "tr": (
        "Böyle zor bir durumda olmanıza çok üzüldüm. Lütfen yalnız kalmayın ve hemen ücretsiz, gizli profesyonel destek hattını arayın:\n\n"
        "• **TelefonSeelsorge Kriz Hattı (7/24 ücretsiz):** `0800 111 0 111`\n"
        "• **Hildesheim Sosyal Psikiyatri Servisi:** Bischof-Janssen-Str. 31, 31134 Hildesheim | Tel: `05121 309-7371`\n"
        "• **Acil Yardım (7/24):** `112`"
    ),
    "ar": (
        "يؤسفني جداً أنك تمر بهذا الظرف العصيب. أنت لست وحدك، يرجى التواصل فوراً مع جهات الدعم النفسي المجانية والسرية:\n\n"
        "• **خط مساعدة الأزمات (TelefonSeelsorge - مجاني على مدار الساعة):** `0800 111 0 111`\n"
        "• **الخدمة النفسية الاجتماعية في هيلدسهايم:** Bischof-Janssen-Str. 31, 31134 Hildesheim | هاتف: `05121 309-7371`\n"
        "• **طوارئ الإسعاف (24/7):** `112`"
    ),
}

CRISIS_RESPONSE_VIOLENCE: Dict[str, str] = {
    "de": (
        "Wenn Sie von Gewalt, Bedrohung oder sexuellen Übergriffen betroffen sind, gibt es in Hildesheim und bundesweit vertrauliche Schutz- und Hilfsangebote:\n\n"
        "• **Hilfetelefon „Gewalt gegen Frauen“ (24/7, kostenlos & in 18 Sprachen):** `116 016`\n"
        "• **Hilfetelefon Sexueller Missbrauch (anonym & kostenfrei):** `0800 22 55 530`\n"
        "• **Frauenhaus Hildesheim (Schutz & Zuflucht):** Tel.: `05121 15454`\n"
        "• **BISS – Beratungs- und Interventionsstelle bei häuslicher Gewalt Hildesheim:** Tel.: `05121 6971418`\n"
        "• **Polizei-Notruf (bei akuter Gefahr):** `110`\n\n"
        "*(Aus Diskretionsgründen bleibt die Sprachausgabe am Terminal stumm. Scannen Sie die Kontakte gerne per QR-Code.)*"
    ),
}

ADDICTION_SAFETY_NET_NOTE = (
    "\n\n---\n"
    "🛡️ **Verifizierter Direktkontakt für Sucht- & Drogenberatung in Hildesheim:**\n"
    "• **DROBS Hildesheim (Paritätische Suchthilfe):** Kaiserstraße 41, 31134 Hildesheim | Tel.: `05121 69605-0`\n"
    "• **Sucht- & Drogen-Hotline (24/7):** `01806 313031` (Bei akuter Vergiftung/Überdosis bitte sofort den **Notruf `112`** wählen!)"
)


def evaluate_safety_and_crisis(
    req: ChatRequest, text_to_check: str
) -> Tuple[Optional[ChatResponse], Optional[str], Optional[AgentTraceStep]]:
    """
    Prüft jede Nutzereingabe deterministisch VOR der RAG-Suche:
    - Gibt bei Stufe 1 (Suizid/Gewalt) oder Stufe 3 (NSFW-Trolling) sofort eine fertige ChatResponse zurück (Hard Exit).
    - Gibt bei Stufe 2 (Drogen/Sucht) einen Zusatz-Hinweis zurück, der an die Antwort angehängt wird.
    """
    combined = f"{req.user_message} {text_to_check}".lower()

    # -----------------------------------------------------------------
    # STUFE 1A: AKUTE SUIZIDALITÄT / SELBSTGEFÄHRDUNG -> HARD INTERRUPT
    # -----------------------------------------------------------------
    if any(re.search(pat, combined, re.IGNORECASE) for pat in SUICIDE_TRIGGERS):
        msg = CRISIS_RESPONSE_SUICIDE.get(req.user_language, CRISIS_RESPONSE_SUICIDE["de"])
        vcard = (
            "BEGIN:VCARD\nVERSION:3.0\n"
            "FN:TelefonSeelsorge & Krisenhilfe Hildesheim\n"
            "TEL:08001110111\n"
            "ADR:;;Bischof-Janssen-Str. 31;Hildesheim;;31134;\n"
            "NOTE:Sozialpsychiatrischer Dienst Hildesheim: 05121 309-7371 | Notruf: 112\n"
            "END:VCARD"
        )
        step = AgentTraceStep(
            step_name="Safety- & Crisis-Guardrail (Stufe 1: Akute Krise / FA-06)",
            details="Trigger für Suizidalität/Selbstgefährdung erkannt -> Hard Interrupt (Kein LLM-Aufruf, Ausgabe verifizierter Notfallkontakte).",
            duration_ms=0.5,
        )
        return (
            ChatResponse(
                session_id=req.session_id,
                paradigm_used=req.paradigm,
                translation_variant_used=req.translation_variant,
                answer_text=msg,
                rewritten_query="[KRISE-INTERVENTION: SUIZIDPRÄVENTION]",
                passed_source_guard=True,
                is_sensitive_topic=True,
                sources=[],
                qr_code_payload=vcard,
                robot_state=RobotState.IDLE,
                agent_trace=[step],
                total_latency_ms=1.0,
            ),
            None,
            step,
        )

    # -----------------------------------------------------------------
    # STUFE 1B: AKUTE / SEXUALISIERTE GEWALT -> HARD INTERRUPT
    # -----------------------------------------------------------------
    if any(re.search(pat, combined, re.IGNORECASE) for pat in VIOLENCE_ABUSE_TRIGGERS):
        msg = CRISIS_RESPONSE_VIOLENCE.get(req.user_language, CRISIS_RESPONSE_VIOLENCE["de"])
        vcard = (
            "BEGIN:VCARD\nVERSION:3.0\n"
            "FN:Hilfetelefon Gewalt & Frauenhaus Hildesheim\n"
            "TEL:116016\n"
            "NOTE:Frauenhaus Hildesheim: 05121 15454 | BISS Hildesheim: 05121 6971418 | Polizei: 110\n"
            "END:VCARD"
        )
        step = AgentTraceStep(
            step_name="Safety- & Crisis-Guardrail (Stufe 1: Gewaltschutz / FA-06)",
            details="Trigger für physische/sexualisierte Gewalt erkannt -> Hard Interrupt (Ausgabe verifizierter Schutzstellen).",
            duration_ms=0.5,
        )
        return (
            ChatResponse(
                session_id=req.session_id,
                paradigm_used=req.paradigm,
                translation_variant_used=req.translation_variant,
                answer_text=msg,
                rewritten_query="[KRISE-INTERVENTION: GEWALTSCHUTZ]",
                passed_source_guard=True,
                is_sensitive_topic=True,
                sources=[],
                qr_code_payload=vcard,
                robot_state=RobotState.IDLE,
                agent_trace=[step],
                total_latency_ms=1.0,
            ),
            None,
            step,
        )

    # -----------------------------------------------------------------
    # STUFE 3: SEXUALISIERTE / MISSBRÄUCHLICHE EINGABEN (BOUNDARY EXIT)
    # -----------------------------------------------------------------
    if any(re.search(pat, combined, re.IGNORECASE) for pat in NSFW_TROLLING_TRIGGERS):
        step = AgentTraceStep(
            step_name="Safety-Guardrail (Stufe 3: Boundary Exit)",
            details="Unangemessene/sexualisierte Eingabe erkannt -> Sachliche Grenzziehung ohne LLM-Aufruf.",
            duration_ms=0.3,
        )
        return (
            ChatResponse(
                session_id=req.session_id,
                paradigm_used=req.paradigm,
                translation_variant_used=req.translation_variant,
                answer_text=(
                    "Ich bin ein digitaler Wegweiser für soziale Beratungs- und Unterstützungsangebote "
                    "im Wohnquartier Hildesheim. Unangemessene oder sexualisierte Anfragen beantworte ich nicht. "
                    "Wenn Sie eine Frage zu lokalen Hilfsangeboten haben, helfe ich Ihnen gerne sachlich weiter."
                ),
                rewritten_query="[BLOCKIERT: UNANGEMESSENE ANFRAGE]",
                passed_source_guard=False,
                is_sensitive_topic=True,
                sources=[],
                qr_code_payload=None,
                robot_state=RobotState.IDLE,
                agent_trace=[step],
                total_latency_ms=1.0,
            ),
            None,
            step,
        )

    # -----------------------------------------------------------------
    # STUFE 2: DROGEN / SUCHT -> RAG ERLAUBEN + FESTES SICHERHEITSNETZ
    # -----------------------------------------------------------------
    if any(re.search(pat, combined, re.IGNORECASE) for pat in DRUG_EMERGENCY_TRIGGERS):
        step = AgentTraceStep(
            step_name="Safety-Guardrail (Stufe 2: Sucht-Sicherheitsnetz & Diskretion SA-06)",
            details="Thema Sucht/Drogen erkannt -> Diskretions-Modus aktiv + verifizierter DROBS-Hildesheim-Kontakt angehängt.",
            duration_ms=0.4,
        )
        return None, ADDICTION_SAFETY_NET_NOTE, step

    return None, None, None