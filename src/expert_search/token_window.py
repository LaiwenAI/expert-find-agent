"""Conservative local rolling-token ledger for controlled API measurements."""
import math
from collections import deque


class TokenWindow:
    def __init__(self,budget=225000,seconds=60.):
        if budget<=0 or not math.isfinite(seconds) or seconds<=0:raise ValueError('Invalid token window')
        self.budget=budget;self.seconds=seconds;self.events=deque()

    def record(self,when,tokens):
        if not math.isfinite(when) or not math.isfinite(tokens) or tokens<0:raise ValueError('Invalid usage event')
        if self.events and when<self.events[-1][0]:raise ValueError('Usage events out of order')
        self.events.append((when,tokens))

    def delay(self,now,estimated_tokens):
        if not math.isfinite(now) or not math.isfinite(estimated_tokens) or not 0<=estimated_tokens<=self.budget:
            raise ValueError('Request exceeds measurement token budget')
        while self.events and self.events[0][0]+self.seconds<=now:self.events.popleft()
        used=sum(tokens for _,tokens in self.events)
        if used+estimated_tokens<=self.budget:return 0.
        for when,tokens in self.events:
            used-=tokens
            if used+estimated_tokens<=self.budget:return max(0.,when+self.seconds-now)
        return 0.
