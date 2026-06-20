# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 12:05:54 2023

@author: mahfouz
"""

class Element:
    'Abstract Base Class for model Elements'
    def __init__(self, element_type):
        super().__init__()
        self.element_type = element_type
        self.input = []
        
    def reset(self):
        pass
    
    
    def conv_str_to_number(self, string):
        if string[0] == '-':
            sign = -1
            string = string[1:]
        else:
            sign = 1
        number = 0
        for i in range(len(string)):
            digit = string[len(string) - 1 - i]
            number = number + (ord(digit) - ord('0')) * pow(10, i)
        number = sign * number
        return  number 
      
        
class State(Element):
    'Abstract Base Class for states'

    def __init__(self, element_type, state_name, state_number, state_type, state_desirability):
        Element.__init__(self, element_type)
        self.state_name   = state_name
        self.state_number = self.conv_str_to_number(state_number)
        self.state_type   = state_type
        self.state_desirability = self.conv_str_to_number(state_desirability)
        self.input = []
                
class Edge(Element):
    'Abstract Base Class for edges'

    def __init__(self, element_type, source_state,  target_state, prob_value):
        Element.__init__(self, element_type)
        self.source_state = self.conv_str_to_number(source_state)
        self.target_state = self.conv_str_to_number(target_state)
        self.prob_value   = float(prob_value)

