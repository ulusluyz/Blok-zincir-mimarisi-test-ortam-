import pytest
from src.audit_key_store import provision_audit_key
from src.audit_rotation import rotate_audit_epoch
from src.blocks import AccessGrantManager


def make_old(tmp_path):
    key = tmp_path/'old.key'
    secret = provision_audit_key(key)
    db, anchor = tmp_path/'old.sqlite', tmp_path/'old.anchor'
    mgr = AccessGrantManager(secret_key=secret, audit_db_path=str(db), audit_anchor_path=str(anchor))
    event = {'event': 'initial'}
    event['audit_signature'] = mgr.sign_audit(event)
    mgr.append_audit(event)
    mgr._audit_db.close()
    return key, db, anchor


def test_rotation_preserves_both_signed_epochs(tmp_path):
    key, db, anchor = make_old(tmp_path)
    new_key = tmp_path/'new.key'
    provision_audit_key(new_key)
    new_db, new_anchor = tmp_path/'new.sqlite', tmp_path/'new.anchor'
    result = rotate_audit_epoch(db, anchor, key, new_db, new_anchor, new_key)
    assert result['previous_tip'][0] == 2
    assert result['new_tip'][0] == 1
    for k,d,a in [(key,db,anchor),(new_key,new_db,new_anchor)]:
        mgr=AccessGrantManager(secret_key=k.read_bytes(), audit_db_path=str(d), audit_anchor_path=str(a))
        assert mgr.verify_audit_chain()
        mgr._audit_db.close()


def test_rotation_rejects_same_key(tmp_path):
    key, db, anchor = make_old(tmp_path)
    with pytest.raises(ValueError, match='differ'):
        rotate_audit_epoch(db,anchor,key,tmp_path/'new.sqlite',tmp_path/'new.anchor',key)


def test_rotation_refuses_existing_destination(tmp_path):
    key, db, anchor = make_old(tmp_path)
    new_key=tmp_path/'new.key'
    provision_audit_key(new_key)
    (tmp_path/'new.sqlite').write_bytes(b'existing')
    with pytest.raises(FileExistsError):
        rotate_audit_epoch(db,anchor,key,tmp_path/'new.sqlite',tmp_path/'new.anchor',new_key)


def test_rotation_rejects_tampered_old_anchor(tmp_path):
    key, db, anchor = make_old(tmp_path)
    new_key=tmp_path/'new.key'
    provision_audit_key(new_key)
    anchor.write_text('tampered')
    with pytest.raises((ValueError,TypeError)):
        rotate_audit_epoch(db,anchor,key,tmp_path/'new.sqlite',tmp_path/'new.anchor',new_key)


def test_resume_rotation_after_signed_handoff(tmp_path):
    from src.audit_rotation import resume_interrupted_rotation
    key, db, anchor = make_old(tmp_path)
    new_key = tmp_path/'new.key'
    provision_audit_key(new_key)
    old = AccessGrantManager(secret_key=key.read_bytes(), audit_db_path=str(db), audit_anchor_path=str(anchor))
    tip = old._ledger_tip()
    event = {'event':'AUDIT_KEY_ROTATION', 'old_tip':tip,
             'new_epoch':'new.sqlite', 'new_anchor':'new.anchor'}
    event['audit_signature'] = old.sign_audit(event)
    old.append_audit(event)
    old._audit_db.close()
    result = resume_interrupted_rotation(db,anchor,key,tmp_path/'new.sqlite',tmp_path/'new.anchor',new_key)
    assert result['previous_tip'][0] == 2
    assert result['new_tip'][0] == 1
    with pytest.raises(FileExistsError):
        resume_interrupted_rotation(db,anchor,key,tmp_path/'new.sqlite',tmp_path/'new.anchor',new_key)


def test_resume_refuses_missing_handoff(tmp_path):
    from src.audit_rotation import resume_interrupted_rotation
    key, db, anchor = make_old(tmp_path)
    new_key = tmp_path/'new.key'
    provision_audit_key(new_key)
    with pytest.raises(ValueError, match='handoff'):
        resume_interrupted_rotation(db,anchor,key,tmp_path/'new.sqlite',tmp_path/'new.anchor',new_key)
