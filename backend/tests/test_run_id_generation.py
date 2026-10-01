import pytest
import uuid
from app.core.database import Run, Container
from app.services import RunService
from datetime import datetime


def test_generate_run_id_format(db_session):
    # Ensure clean runs table for test isolation
    db_session.query(Run).delete()
    db_session.commit()

    service = RunService(db_session)
    date_str = datetime.now().strftime("%d%m%y")

    # Initial run for the day
    run_id_1 = service.generate_run_id()
    assert run_id_1.startswith(f"{date_str}-")
    assert run_id_1 == f"{date_str}-A01"

    # Create dummy container
    container = Container(
        container_type=f"TEST-{uuid.uuid4()}",
        internal_length_cm=1000.0,
        internal_width_cm=200.0,
        internal_height_cm=200.0,
        max_weight_kg=20000.0,
    )
    db_session.add(container)
    db_session.commit()
    db_session.refresh(container)

    # Simulate run A01 exists
    run_1 = Run(run_id=run_id_1, container_id=container.id, packing_list_json="{}", status="completed")
    db_session.add(run_1)
    db_session.commit()

    run_id_2 = service.generate_run_id()
    assert run_id_2 == f"{date_str}-A02"

    # Simulate run A99 exists
    run_99 = Run(run_id=f"{date_str}-A99", container_id=container.id, packing_list_json="{}", status="completed")
    db_session.add(run_99)
    db_session.commit()

    # Next run should rollover to B01
    run_id_100 = service.generate_run_id()
    assert run_id_100 == f"{date_str}-B01"

    # Simulate run B99 exists
    run_198 = Run(run_id=f"{date_str}-B99", container_id=container.id, packing_list_json="{}", status="completed")
    db_session.add(run_198)
    db_session.commit()

    # Next run should rollover to C01
    run_id_199 = service.generate_run_id()
    assert run_id_199 == f"{date_str}-C01"
