========================================================================
HOMIE - QUARTIERS-CHATBOT HILDESHEIM (IT-MA5 / HULLS PROJEKT)
========================================================================

Lokaler, mehrsprachiger und sprachgesteuerter Quartiers-Chatbot auf Basis
einer Microservice-Architektur (Quality-Gate, ETL, Hybrid-RAG, Orchestrator,
Speech-Service & Gradio-Steuerzentrale).


1. VORAUSSETZUNGEN (WINDOWS 10 / 11)
------------------------------------------------------------------------
- Python 3.10 bis 3.12 (mit PATH-Eintrag installiert)
- Git (https://git-scm.com/)
- Ollama für Windows (https://ollama.com/download)
- Internetverbindung beim allerersten Start (zum Herunterladen der lokalen
  KI-, Embedding- und Sprachmodelle)


2. INSTALLATION & DOWNLOAD
------------------------------------------------------------------------
1) Repository klonen und in den Projektordner wechseln (PowerShell):

   git clone https://github.com/JamesMarcel12/homie-quartiers-chatbot.git
   cd homie-quartiers-chatbot

2) Optional: Konfigurationsdatei anlegen (Standardwerte funktionieren
   auch direkt ohne Anpassung):

   copy .env.example .env

3) Python-Abhängigkeiten:
   Das Startskript installiert benötigte Abhängigkeiten bzw. nutzt die
   virtuelle Umgebung (.venv). Falls Pakete manuell nachinstalliert
   werden müssen:

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements-dev.txt


3. OLLAMA STARTEN & SPRACHMODELL LADEN (WICHTIG!)
------------------------------------------------------------------------
1) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   Unter Windows muss die installierte Ollama-Desktop-App zwingend über
   das Windows-Startmenü GEÖFFNET werden (das Ollama-Symbol muss unten
   rechts in der Taskleiste sichtbar sein). Ein reiner Terminal-Befehl
   reicht unter Windows häufig nicht aus. 
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

2) Einmalig das lokale Sprachmodell in der PowerShell herunterladen:

   ollama pull qwen2.5:7b-instruct


4. ANWENDUNG STARTEN (POWERSHELL)

Im Projektordner (./SprachBot/) eine PowerShell (vllt. Admin) öffnen und folgenden Befehl ausführen:

   .\start_dev_stack.ps1

Hinweis: Falls Windows die Ausführung von PowerShell-Skripten blockiert,
vorher einmalig für das aktuelle Fenster freigeben:

   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\start_dev_stack.ps1

Das Skript startet alle Microservices automatisch:
- Quality-Gate Service : Port 8001
- ETL-Ingest Service   : Port 8002
- RAG-Retrieval        : Port 8003
- Orchestrator         : Port 8004
- Speech Service       : Port 8005
- Gradio UI (Browser)  : http://127.0.0.1:7860

Wichtiger Hinweis zum ersten Start:
Beim ersten Start werden die lokalen Modelle (BGE-M3 Embedding,
BGE-Reranker, faster-whisper & Piper-TTS) automatisch heruntergeladen.
Dies dauert je nach Internetverbindung einige Minuten.

In der Gradio-Oberfläche unter "Tab 3: Microservice Health-Monitor"
lässt sich mit einem Klick prüfen, ob alle Services (inkl. Ollama auf
Port 11434) grün / ONLINE sind.


5. BEISPIELDOKUMENTE (ORDNER "papers/") & DATENBANK
------------------------------------------------------------------------
Im Ordner "papers/" sind bereits Beispieldokumente (PDFs) aus dem Raum
Hildesheim hinterlegt.

Da lokale Vektordatenbanken in der Regel nicht vollständig über Git
versioniert werden, prüfen Sie nach dem Start bitte kurz "Tab 2:
Quality-Gate, Massen-Upload & Inventar":

- Sind oben in der Inventar-Tabelle bereits Dokumente gelistet, ist der
  Chatbot in Tab 1 sofort einsatzbereit.
- Ist die Inventar-Tabelle noch leer:
  1) In Tab 2 unter "A. Datei(en) hochladen" alle PDFs aus dem Ordner
     "papers/" markieren und gemeinsam hineinziehen (Massen-Upload).
  2) Sicherstellen, dass der Haken "Nach Prüfung automatisch in ChromaDB
     & BM25 übernehmen" gesetzt ist.
  3) Auf "1. Datei(en) prüfen (Quality-Gate) & einlesen" klicken.
  Nach wenigen Sekunden sind alle Dokumente geprüft, gechunkt und im
  Chatbot (Tab 1) abrufbar.


6. BEDIENUNG & FUNKTIONSÜBERSICHT
------------------------------------------------------------------------
- Tab 1 (Kiosk-Terminal): Chat in Text- und Sprachform (DE, TR, AR, UK,
  RU, EN), Live-Umschaltung zwischen Agentic AI / Pure RAG / Fine-Tuned,
  Sprachniveau (Standard / Einfach / Leichte Sprache), V2a vs. V2b,
  QR-Code-Ausgabe und Live-Agent-Trace.
- Tab 2 (Quality-Gate & ETL): Einzel-/Massen-Upload von PDFs, Web-Scraper
  mit Blacklist- & Relevanzprüfung, Audit-Protokoll (§ 1 - § 11) und
  Wissensbasis-Inventar.
- Tab 3 (Health-Monitor): Statusprüfung aller lokalen Dienste.

Tipp: In der gesamten Oberfläche öffnen die blauen "i"-Symbole beim
Anklicken eine kurze Erklärung zur jeweiligen Einstellung.


7. PROJEKTSTRUKTUR
------------------------------------------------------------------------
config/                Regelwerk-Konfigurationen (Blacklist, Glossar,
                       Abkürzungen, Antidiskriminierungs-Liste)
data/                  Lokale Datenablage (ChromaDB, BM25, Audit-Logs,
                       Quarantäne, Modell-Cache)
papers/                Beispieldokumente (PDFs) zum Testen
services/              Microservices (Quality-Gate, ETL, RAG,
                       Orchestrator, Speech, Gradio UI)
shared/                Gemeinsame Pydantic-Schemas & Logger
requirements-dev.txt   Python-Abhängigkeiten für Entwicklung/Evaluation
requirements-robot.txt Abhängigkeiten für den Robotik-Betrieb
start_dev_stack.ps1    PowerShell-Startskript für den gesamten Stack