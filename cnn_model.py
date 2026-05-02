"""
FirefighterDispatcher
---------------------
Routes classified transmissions to the appropriate response action,
notifying the relevant units based on urgency level.
"""

import datetime
import logging

logger = logging.getLogger(__name__)

DISPATCH_PROTOCOLS = {
    3: {
        "action": "IMMEDIATE MAYDAY RESPONSE — Alert all units and incident commander",
        "notify": ["Incident Commander", "Rapid Intervention Team", "EMS"],
        "alert_tone": "emergency_tone_3.wav",
    },
    2: {
        "action": "PRIORITY DISPATCH — Notify sector commander and request backup",
        "notify": ["Sector Commander", "Backup Engine", "EMS Standby"],
        "alert_tone": "alert_tone_2.wav",
    },
    1: {
        "action": "RESOURCE REQUEST — Log and relay to logistics officer",
        "notify": ["Logistics Officer", "Supply Unit"],
        "alert_tone": "alert_tone_1.wav",
    },
    0: {
        "action": "ROUTINE LOG — Record transmission in incident log",
        "notify": [],
        "alert_tone": None,
    },
}


class FirefighterDispatcher:
    def dispatch(self, urgency_result: dict, transcription: str) -> dict:
        level = urgency_result["level"]
        protocol = DISPATCH_PROTOCOLS[level]
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"

        log_entry = {
            "timestamp": timestamp,
            "transcription": transcription,
            "urgency_level": level,
            "urgency_label": urgency_result["label"],
            "action": protocol["action"],
            "notified_units": protocol["notify"],
            "alert_tone": protocol["alert_tone"],
        }

        logger.info(f"[DISPATCH] {protocol['action']}")
        if protocol["notify"]:
            logger.info(f"[NOTIFY]   {', '.join(protocol['notify'])}")

        return log_entry
