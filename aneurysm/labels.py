LOCATIONS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
]
LABELS = LOCATIONS + ['Aneurysm Present']
MODALITIES = ['CTA', 'MRA', 'MRI T2', 'MRI T1post']
FLIP_LR = [1, 0, 3, 2, 5, 4, 6, 8, 7, 10, 9, 11, 12]
N_LOC = len(LOCATIONS)
N_TERR = N_LOC + 1
TERR_FLIP = [0] + [k + 1 for k in FLIP_LR]
