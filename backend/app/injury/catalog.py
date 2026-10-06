"""Static knowledge used by the simulator, the API and the UI.

Every numeric prior below is tagged with the source(s) that *informed* it. Sources
were read as PubMed abstracts; the exact numbers in the sport/region tables are our
own assumptions shaped by those findings unless the comment quotes a figure.
"""
from __future__ import annotations

REGIONS = [
    "hamstring", "quadriceps", "groin", "calf_achilles", "knee", "ankle",
    "foot", "hip", "lower_back", "shoulder", "elbow_wrist", "head_neck",
]

REGION_LABELS = {
    "hamstring": "Hamstring",
    "quadriceps": "Quadriceps",
    "groin": "Groin / adductor",
    "calf_achilles": "Calf / Achilles",
    "knee": "Knee (ACL/MCL, meniscus, patellar)",
    "ankle": "Ankle",
    "foot": "Foot",
    "hip": "Hip",
    "lower_back": "Lower back",
    "shoulder": "Shoulder",
    "elbow_wrist": "Elbow / wrist / hand",
    "head_neck": "Head / neck (incl. concussion)",
}

INJURY_TYPES = ["muscle_strain", "ligament_sprain", "tendinopathy", "overuse_stress", "contusion", "concussion"]
TYPE_LABELS = {
    "muscle_strain": "Muscle strain",
    "ligament_sprain": "Ligament sprain",
    "tendinopathy": "Tendinopathy",
    "overuse_stress": "Overuse / bone stress",
    "contusion": "Contusion",
    "concussion": "Concussion",
}

# Which injury types are plausible per region, with base weights (renormalised in the simulator).
REGION_TYPES = {
    "hamstring": {"muscle_strain": 0.85, "tendinopathy": 0.10, "contusion": 0.05},
    "quadriceps": {"muscle_strain": 0.55, "contusion": 0.35, "tendinopathy": 0.10},
    "groin": {"muscle_strain": 0.55, "tendinopathy": 0.30, "overuse_stress": 0.15},
    "calf_achilles": {"muscle_strain": 0.45, "tendinopathy": 0.35, "overuse_stress": 0.20},
    "knee": {"ligament_sprain": 0.45, "tendinopathy": 0.30, "overuse_stress": 0.15, "contusion": 0.10},
    "ankle": {"ligament_sprain": 0.80, "tendinopathy": 0.10, "contusion": 0.10},
    "foot": {"overuse_stress": 0.60, "ligament_sprain": 0.20, "contusion": 0.20},
    "hip": {"tendinopathy": 0.35, "muscle_strain": 0.30, "overuse_stress": 0.20, "contusion": 0.15},
    "lower_back": {"overuse_stress": 0.55, "muscle_strain": 0.45},
    "shoulder": {"tendinopathy": 0.50, "ligament_sprain": 0.35, "contusion": 0.15},
    "elbow_wrist": {"ligament_sprain": 0.50, "tendinopathy": 0.30, "contusion": 0.20},
    "head_neck": {"concussion": 0.50, "contusion": 0.35, "ligament_sprain": 0.15},
}

SPORTS = ["Athletics", "Kabaddi", "Kho-Kho", "Football", "Hockey", "Volleyball", "Badminton", "Wrestling"]
CONTACT_SPORTS = {"Kabaddi", "Football", "Hockey", "Wrestling"}
TEAM_SPORTS = {"Kabaddi", "Kho-Kho", "Football", "Hockey", "Volleyball"}
OVERHEAD_SPORTS = {"Volleyball", "Badminton"}
RUNNING_SPORTS = {"Athletics", "Football", "Hockey", "Kho-Kho"}

# Relative share of injuries by region for each sport (normalised in code).
SPORT_REGION_PRIOR = {
    # Thigh strain is the most common subtype in pro football (17%, Ekstrand 2011); lower-extremity
    # and muscle/tendon injuries dominate (López-Valenciano 2020); ankle common in team sports (Fong 2007).
    "Football": dict(hamstring=.16, quadriceps=.07, groin=.10, calf_achilles=.07, knee=.15, ankle=.15, foot=.05,
                     hip=.04, lower_back=.04, shoulder=.03, elbow_wrist=.03, head_neck=.06),
    # Thigh strain 28.2% of diagnoses, then lower-leg strain and ankle sprain (Feddermann-Demont 2014);
    # knee is the predominant site in distance runners (van Gent 2007).
    "Athletics": dict(hamstring=.24, quadriceps=.05, groin=.05, calf_achilles=.14, knee=.12, ankle=.10, foot=.10,
                      hip=.06, lower_back=.07, shoulder=.02, elbow_wrist=.02, head_neck=.03),
    # Contact raid/tackle sport; orofacial/head trauma highly prevalent in youth kabaddi (Johnson 2023).
    # Remaining split is an assumption (contact-sport pattern, Hammer 2020).
    "Kabaddi": dict(hamstring=.07, quadriceps=.05, groin=.06, calf_achilles=.04, knee=.20, ankle=.13, foot=.03,
                    hip=.04, lower_back=.06, shoulder=.12, elbow_wrist=.06, head_neck=.14),
    # No epidemiology found for Kho-Kho: assumption based on sprint / sudden-dodge demands.
    "Kho-Kho": dict(hamstring=.14, quadriceps=.06, groin=.07, calf_achilles=.08, knee=.16, ankle=.20, foot=.07,
                    hip=.04, lower_back=.05, shoulder=.03, elbow_wrist=.05, head_neck=.05),
    # Ankle sprain is the major ankle injury in field hockey (Fong 2007); the rest is an assumption.
    "Hockey": dict(hamstring=.10, quadriceps=.05, groin=.07, calf_achilles=.05, knee=.14, ankle=.18, foot=.05,
                   hip=.05, lower_back=.10, shoulder=.03, elbow_wrist=.06, head_neck=.12),
    # Ankle sprains, patellar tendinopathy, finger/thumb sprains, shoulder overuse, concussion
    # (Young 2023); ankle most common site in indoor volleyball (Fong 2007).
    "Volleyball": dict(hamstring=.03, quadriceps=.03, groin=.02, calf_achilles=.03, knee=.18, ankle=.24, foot=.03,
                       hip=.02, lower_back=.10, shoulder=.18, elbow_wrist=.10, head_neck=.04),
    # Assumption: overhead (Tooth 2020) + court-sport (Fong 2007) pattern.
    "Badminton": dict(hamstring=.04, quadriceps=.03, groin=.02, calf_achilles=.12, knee=.18, ankle=.16, foot=.06,
                      hip=.02, lower_back=.10, shoulder=.16, elbow_wrist=.10, head_neck=.01),
    # Head/face is the most common site in boys' high-school contact sports (22.5%) and the knee
    # the most common site of severe injury (Hammer 2020); remaining split is an assumption.
    "Wrestling": dict(hamstring=.03, quadriceps=.03, groin=.04, calf_achilles=.02, knee=.18, ankle=.07, foot=.01,
                      hip=.04, lower_back=.08, shoulder=.18, elbow_wrist=.10, head_neck=.22),
}

# General guidance shown in the UI. Not medical advice.
PREVENTION = {
    "hamstring": {
        "prevention": ["Nordic hamstring curls: progressive eccentric programme, then 1-2x/week in season",
                       "Build sprint exposure gradually; avoid big jumps in high-speed volume",
                       "Track 30m fly / jump drops as a fatigue flag"],
        "return_to_play": ["Pain-free isometrics → eccentric loading → progressive running to full sprint",
                           "Return only after full-speed sprint and strength close to the other leg"],
        "evidence": "Nordic hamstring programme cut acute hamstring injuries (rate ratio 0.29) in a cluster RCT of 942 soccer players (Petersen 2011).",
    },
    "quadriceps": {
        "prevention": ["Progressive kicking / sprint volume", "Eccentric quad work (reverse Nordics, split squats)"],
        "return_to_play": ["Restore knee flexion range, then graded kicking/sprinting"],
        "evidence": "General guidance (no specific trial read).",
    },
    "groin": {
        "prevention": ["Copenhagen adduction exercise, 3 levels, 3x/week pre-season then 1x/week",
                       "Limit sudden increases in change-of-direction and kicking volume"],
        "return_to_play": ["Pain-free adductor squeeze, then graded change-of-direction and kicking"],
        "evidence": "Adductor Strengthening Programme (Copenhagen adduction) lowered groin-problem risk by 41% (OR 0.59) in a cluster RCT (Harøy 2019).",
    },
    "calf_achilles": {
        "prevention": ["Heavy slow calf raises (straight + bent knee)", "Avoid sudden jumps in weekly running distance",
                       "Monitor footwear and hard-surface exposure"],
        "return_to_play": ["Single-leg calf-raise capacity close to the other side before running; graded hopping"],
        "evidence": "General guidance; long weekly distance is a risk factor in male distance runners (van Gent 2007).",
    },
    "knee": {
        "prevention": ["Neuromuscular warm-up: landing mechanics, single-leg balance, deceleration drills",
                       "Hamstring + glute strength; manage jump volume (patellar tendon)",
                       "Extra attention for female athletes and after a previous knee injury"],
        "return_to_play": ["Criteria-based progression (strength, hop tests, sport-specific drills); physio sign-off after ligament injury"],
        "evidence": "ACL injury is more common in female high-school/college athletes (Lin 2018); a previous knee injury raised same-site risk 2-3x (Hägglund 2006).",
    },
    "ankle": {
        "prevention": ["Balance / proprioception work (single-leg, wobble board)", "Taping or bracing after a previous sprain",
                       "Calf and peroneal strength"],
        "return_to_play": ["Full range, pain-free hopping and cutting before return"],
        "evidence": "Ankle is among the most injured sites in court and team sports, and sprain is the main ankle injury (Fong 2007).",
    },
    "foot": {
        "prevention": ["Gradual mileage progression", "Adequate energy intake (bone health)", "Footwear rotation; check hard surfaces"],
        "return_to_play": ["Pain-free walking/hopping; graded run-walk programme; medical review for suspected stress fracture"],
        "evidence": "Bone stress injuries are more common in female athletes; relative energy deficiency is a risk factor (Lin 2018).",
    },
    "hip": {
        "prevention": ["Glute and hip-abductor strength", "Mobility work; manage sprint and change-of-direction load"],
        "return_to_play": ["Pain-free single-leg tasks, then graded sport drills"],
        "evidence": "General guidance (no specific trial read).",
    },
    "lower_back": {
        "prevention": ["Trunk endurance (planks, side planks, bird-dogs)", "Limit repetitive hyperextension volume in growing athletes",
                       "Monitor growth spurts in youth"],
        "return_to_play": ["Pain-free trunk control, graded return of extension/rotation loads; medical review for persistent pain in youth"],
        "evidence": "Around peak height velocity young players face more overuse / growth-related injury (Towlson 2021).",
    },
    "shoulder": {
        "prevention": ["Rotator-cuff strengthening (external rotation)", "Maintain shoulder range of motion",
                       "Manage overhead / throwing / smash volume"],
        "return_to_play": ["Full range and strength, then graded overhead volume"],
        "evidence": "Previous injury, range-of-motion deficits, rotator-cuff weakness and training load are risk factors in overhead athletes (Tooth 2020).",
    },
    "elbow_wrist": {
        "prevention": ["Grip and forearm strength", "Fall technique (contact sports)", "Finger/thumb taping in volleyball"],
        "return_to_play": ["Pain-free grip and weight-bearing; taping/bracing as needed"],
        "evidence": "Finger and thumb sprains are common in volleyball (Young 2023).",
    },
    "head_neck": {
        "prevention": ["Neck strength training", "Safe tackling / falling technique", "Mouthguard in contact play",
                       "Remove from play if a concussion is suspected"],
        "return_to_play": ["Graded return-to-sport protocol after a concussion with medical clearance; never same-day return"],
        "evidence": "Head/face was the most common injury site in boys' high-school contact sports (Hammer 2020); orofacial trauma is highly prevalent in youth kabaddi (Johnson 2023).",
    },
}

SOURCES = [
    {"key": "Foster 1998", "title": "Monitoring training in athletes with reference to overtraining syndrome", "journal": "Med Sci Sports Exerc 1998;30(7):1164-8", "pmid": "9662690",
     "used_for": "Session-RPE load, monotony (daily mean/SD) and strain (load x monotony) definitions; high strain as a risk signal."},
    {"key": "Hulin 2016", "title": "The acute:chronic workload ratio predicts injury: high chronic workload may decrease injury risk in elite rugby league players", "journal": "Br J Sports Med 2016;50(4):231-6", "pmid": "26511006",
     "used_for": "ACWR spikes (>~1.5) raise risk; high chronic load is protective at moderate ratios (0.85-1.35)."},
    {"key": "Milewski 2014", "title": "Chronic lack of sleep is associated with increased sports injuries in adolescent athletes", "journal": "J Pediatr Orthop 2014;34(2):129-33", "pmid": "25028798",
     "used_for": "<8 h sleep associated with 1.7x injury likelihood."},
    {"key": "Hägglund 2006", "title": "Previous injury as a risk factor for injury in elite football", "journal": "Br J Sports Med 2006;40(9):767-72", "pmid": "16855067",
     "used_for": "Previous injury HR 2.7; previous hamstring, groin, knee injuries 2-3x same-site risk; no such relation for ankle sprain."},
    {"key": "Ekstrand 2011", "title": "Injury incidence and injury patterns in professional football: the UEFA injury study", "journal": "Br J Sports Med 2011;45(7):553-8", "pmid": "19553225",
     "used_for": "Match vs training incidence (27.5 vs 4.1/1000 h); thigh strain 17% of injuries; re-injuries 12%."},
    {"key": "López-Valenciano 2020", "title": "Epidemiology of injuries in professional football: a systematic review and meta-analysis", "journal": "Br J Sports Med 2020;54(12):711-8", "pmid": "31171515",
     "used_for": "Lower extremity and muscle/tendon injuries dominate in football."},
    {"key": "Petersen 2011", "title": "Preventive effect of eccentric training on acute hamstring injuries in men's soccer", "journal": "Am J Sports Med 2011;39(11):2296-303", "pmid": "21825112",
     "used_for": "Nordic hamstring programme: acute hamstring injury rate ratio 0.293."},
    {"key": "Harøy 2019", "title": "The Adductor Strengthening Programme prevents groin problems among male football players", "journal": "Br J Sports Med 2019;53(3):150-7", "pmid": "29891614",
     "used_for": "Copenhagen adduction programme: 41% lower groin-problem risk (OR 0.59)."},
    {"key": "Feddermann-Demont 2014", "title": "Injuries in 13 international Athletics championships between 2007-2012", "journal": "Br J Sports Med 2014;48(7):513-22", "pmid": "24620039",
     "used_for": "Athletics: thigh strain 28.2%, then lower-leg strain and ankle sprain."},
    {"key": "van Gent 2007", "title": "Incidence and determinants of lower extremity running injuries in long distance runners", "journal": "Br J Sports Med 2007;41(8):469-80", "pmid": "17473005",
     "used_for": "Knee is the predominant running-injury site; long weekly distance (men) and previous injury are risk factors."},
    {"key": "Fong 2007", "title": "A systematic review on ankle injury and ankle sprain in sports", "journal": "Sports Med 2007;37(1):73-94", "pmid": "17190537",
     "used_for": "Ankle most common site in 24/70 sports incl. indoor volleyball; sprain the main ankle injury incl. field hockey."},
    {"key": "Young 2023", "title": "Epidemiology of Common Injuries in the Volleyball Athlete", "journal": "Curr Rev Musculoskelet Med 2023;16(6):229-34", "doi": "10.1007/s12178-023-09826-2",
     "used_for": "Volleyball: ankle sprains, patellar tendinopathy, finger/thumb sprains, shoulder overuse, concussion."},
    {"key": "Hammer 2020", "title": "Epidemiology of Injuries Sustained in Boys' High School Contact and Collision Sports, 2008-2009 Through 2012-2013", "journal": "Orthop J Sports Med 2020;8(2)", "doi": "10.1177/2325967120903699",
     "used_for": "Competition vs practice RR 4.0; head/face 22.5% most common site; knee most common severe-injury site."},
    {"key": "Johnson 2023", "title": "Prevalence and pattern of traumatic orofacial injuries in Kabaddi players in Delhi-NCR region", "journal": "Injury 2023;54(6):1510-8", "pmid": "36922269",
     "used_for": "Orofacial trauma prevalence 44% in 10-18-year-old kabaddi players."},
    {"key": "Tooth 2020", "title": "Risk Factors of Overuse Shoulder Injuries in Overhead Athletes: A Systematic Review", "journal": "Sports Health 2020;12(5):478-87", "pmid": "32758080",
     "used_for": "Overhead shoulder risk: previous injury, ROM, rotator-cuff weakness, training load."},
    {"key": "Lin 2018", "title": "Sex Differences in Common Sports Injuries", "journal": "PM R 2018;10(10):1073-82", "pmid": "29550413",
     "used_for": "Bone stress injuries and ACL injuries more common in female (high-school/college) athletes."},
    {"key": "Towlson 2021", "title": "Maturity-associated considerations for training load, injury risk, and physical performance in youth soccer", "journal": "J Sport Health Sci 2021;10(4):403-12", "pmid": "32961300",
     "used_for": "Increased overuse / growth-related injury risk around peak height velocity."},
    {"key": "Lövdal 2021 (data)", "title": "Replication Data for: Injury Prediction In Competitive Runners With Machine Learning", "journal": "DataverseNL, doi:10.34894/UWU9PV", "doi": "10.34894/UWU9PV", "licence": "CC0 1.0",
     "mirror": "https://github.com/sonicjoy/Injury-Prediction-for-Competitive-Runners (week_approach_maskedID_timeseries.csv, sha256-checked)",
     "used_for": "REAL training logs of 74 runners (42,798 athlete-days, 575 injury days): trains and validates the real-data load model."},
]
