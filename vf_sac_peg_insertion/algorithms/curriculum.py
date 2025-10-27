"""Curriculum Learning Manager"""

import numpy as np

class CurriculumManager:
    """Manages progressive difficulty curriculum"""
    def __init__(self, config):
        self.config = config
        self.current_stage = 1
        self.stages = config.stages
        self.evaluation_window = config.evaluation_window
        self.recent_success_rates = []
    
    def get_current_stage(self):
        return self.stages[self.current_stage - 1]
    
    def update(self, success):
        self.recent_success_rates.append(float(success))
        
        if len(self.recent_success_rates) > self.evaluation_window:
            self.recent_success_rates.pop(0)
        
        if len(self.recent_success_rates) >= self.evaluation_window:
            avg_success = np.mean(self.recent_success_rates)
            current_stage_config = self.stages[self.current_stage - 1]
            
            if avg_success >= current_stage_config['success_threshold']:
                if self.current_stage < len(self.stages):
                    self.advance_stage()
    
    def advance_stage(self):
        self.current_stage += 1
        self.recent_success_rates = []
        print(f"\n{'='*60}")
        print(f"CURRICULUM ADVANCED TO STAGE {self.current_stage}: {self.get_current_stage()['name'].upper()}")
        print(f"{'='*60}\n")
    
    def is_complete(self):
        return self.current_stage > len(self.stages)
