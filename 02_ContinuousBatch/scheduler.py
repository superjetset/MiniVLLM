from request import Request
from typing import List

class Scheduler:

    waiting_list: List[Request]     # 等待prefill的请求   
    running_list: List[Request]     # 已经prefill，待decode的请求
    req_counter: int

    def __init__(self):
        self.waiting_list: List[Request] = []
        self.running_list: List[Request] = []
        self.req_counter = 0

    def add_request(self, req: Request):
        req.request_id = self.req_counter
        self.waiting_list.append(req) # 新添加的请求直接等待prefill
        self.req_counter += 1

    def remove_request(self, request_id: int)->int:
        for i, req in enumerate(self.running_list):
            if req.request_id == request_id:
                del self.running_list[i]
                return request_id
            
        for i, req in enumerate(self.waiting_list):
            if req.request_id == request_id:
                del self.waiting_list[i]
                return request_id
        return -1
    
    def has_pending(self)->bool:
        return len(self.waiting_list) > 0 or len(self.running_list) > 0
    
    def promote_to_running(self, request_id:int)->int:        
        for i, req in enumerate(self.waiting_list):
            if req.request_id == request_id:
                del self.waiting_list[i]
                self.running_list.append(req)
                return request_id
        return -1