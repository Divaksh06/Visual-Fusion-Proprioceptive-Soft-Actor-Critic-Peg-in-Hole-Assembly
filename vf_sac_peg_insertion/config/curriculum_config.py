"""Curriculum Learning Configuration"""

class CurriculumConfig:
    stages = [
        {
            'name': 'simple',
            'hole_clearance': 0.003,  # 3mm
            'position_noise': 0.005,   # 5mm
            'orientation_noise': 0.0,  # 0 degrees
            'success_threshold': 0.80,
            'episodes': 50000,
        },
        {
            'name': 'moderate',
            'hole_clearance': 0.001,  # 1mm
            'position_noise': 0.010,   # 10mm
            'orientation_noise': 0.05,  # ~3 degrees
            'success_threshold': 0.75,
            'episodes': 75000,
        },
        {
            'name': 'complex',
            'hole_clearance': 0.0002,  # 0.2mm (tight fit)
            'position_noise': 0.015,    # 15mm
            'orientation_noise': 0.10,  # ~6 degrees
            'success_threshold': 0.70,
            'episodes': 100000,
        },
    ]
    
    evaluation_window = 100
