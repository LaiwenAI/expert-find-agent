"""In-memory exposure and delayed-feedback registry for experimental routing."""
import math


class FeedbackEvents:
    """Bind one final numeric outcome to each logged routing exposure.

    Missing outcomes remain pending. Duplicate identical operations are
    idempotent; conflicting duplicates fail. Propensity is logged but is not
    used to debias the router. The caller must persist records externally if
    durability is needed; this registry is not a production event store.
    """
    def __init__(self, router):
        from .feedback_router import FeedbackRouter
        if not isinstance(router, FeedbackRouter):
            raise ValueError('Require an experimental feedback router')
        self.router=router
        self._events={}

    def expose(self, event_id, context, arm, propensity):
        if not isinstance(event_id,str) or not event_id or not isinstance(context,str) or not context or arm not in self.router.arms:
            raise ValueError('Invalid exposure')
        if type(propensity) not in (int,float) or not math.isfinite(propensity) or not 0 < propensity <= 1:
            raise ValueError('Invalid propensity')
        exposure=(context,arm,float(propensity))
        previous=self._events.get(event_id)
        if previous is not None:
            if previous['exposure'] != exposure:
                raise ValueError('Conflicting exposure ID')
            return False
        self._events[event_id]={'exposure':exposure,'outcome':None}
        return True

    def outcome(self, event_id, value):
        event=self._events.get(event_id)
        if event is None:
            raise ValueError('Unknown exposure')
        if value is None:
            return False
        if type(value) not in (int,float) or not math.isfinite(value):
            raise ValueError('Invalid outcome')
        value=float(value)
        if event['outcome'] is not None:
            if event['outcome'] != value:
                raise ValueError('Conflicting final outcome')
            return False
        context,arm,_=event['exposure']
        self.router.observe(context,arm,value)
        event['outcome']=value
        return True

    def snapshot(self):
        return [{'event_id':key,'context':v['exposure'][0],'arm':v['exposure'][1],
                 'propensity':v['exposure'][2],'outcome':v['outcome']}
                for key,v in sorted(self._events.items())]
