"""Same arm-blind witness screen for all four factorial arms."""
from pathlib import Path
from audit_concrete_witnesses import run

if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env')
    run(data=Path('artifacts/factorial-partial-comparison-v1'),root=Path('artifacts/factorial-witness-audit-v1'),arms=('base','language','bridge','both'))
