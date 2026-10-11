"""Rotate audit signing keys by opening a new immutable ledger epoch.

The old epoch is retained and must remain verifiable with its old key.
This is not an in-place re-signing operation.
"""
from __future__ import annotations

from pathlib import Path
from .blocks import AccessGrantManager
from .audit_key_store import load_audit_key


def rotate_audit_epoch(old_db, old_anchor, old_key_path,
                       new_db, new_anchor, new_key_path):
    """Create a linked new epoch. Never overwrite an existing target.

    The old epoch receives a signed handoff entry. A crash between handoff
    and new epoch creation requires operator reconciliation, not silent retry.
    """
    old_db, old_anchor, new_db, new_anchor = map(Path, (old_db, old_anchor, new_db, new_anchor))
    if old_db.resolve() == new_db.resolve() or old_anchor.resolve() == new_anchor.resolve():
        raise ValueError('Old and new epoch paths must differ')
    if new_db.exists() or new_anchor.exists():
        raise FileExistsError('New epoch must not already exist')
    old_key = load_audit_key(old_key_path)
    new_key = load_audit_key(new_key_path)
    if old_key == new_key:
        raise ValueError('New signing key must differ from old key')
    old = AccessGrantManager(secret_key=old_key, audit_db_path=str(old_db), audit_anchor_path=str(old_anchor))
    try:
        if not old.verify_audit_chain():
            raise ValueError('Old epoch failed integrity verification')
        old_tip = old._ledger_tip()
        transition = {'event': 'AUDIT_KEY_ROTATION', 'old_tip': old_tip,
                      'new_epoch': new_db.name, 'new_anchor': new_anchor.name}
        transition['audit_signature'] = old.sign_audit(transition)
        old.append_audit(transition)
        final_old_tip = old._ledger_tip()
        new = AccessGrantManager(secret_key=new_key, audit_db_path=str(new_db), audit_anchor_path=str(new_anchor))
        try:
            start = {'event': 'AUDIT_EPOCH_START', 'previous_tip': final_old_tip,
                     'previous_epoch': old_db.name, 'previous_anchor': old_anchor.name}
            start['audit_signature'] = new.sign_audit(start)
            new.append_audit(start)
            if not old.verify_audit_chain() or not new.verify_audit_chain():
                raise ValueError('Epoch integrity failed after rotation')
            return {'previous_tip': final_old_tip, 'new_tip': new._ledger_tip()}
        finally:
            new._audit_db.close()
    finally:
        old._audit_db.close()


def resume_interrupted_rotation(old_db, old_anchor, old_key_path,
                                new_db, new_anchor, new_key_path):
    """Complete a verified old-epoch handoff without appending it twice.

    Operator-initiated only; refuses partial destinations and altered handoffs.
    """
    import json
    old_db, old_anchor, new_db, new_anchor = map(Path, (old_db, old_anchor, new_db, new_anchor))
    if old_db.resolve() == new_db.resolve() or old_anchor.resolve() == new_anchor.resolve():
        raise ValueError('Epoch paths must differ')
    if new_db.exists() or new_anchor.exists():
        raise FileExistsError('Partial or existing destination requires manual investigation')
    old_key, new_key = load_audit_key(old_key_path), load_audit_key(new_key_path)
    if old_key == new_key:
        raise ValueError('New signing key must differ')
    old = AccessGrantManager(secret_key=old_key, audit_db_path=str(old_db), audit_anchor_path=str(old_anchor))
    try:
        if not old.verify_audit_chain():
            raise ValueError('Old ledger failed verification')
        row = old._audit_db.execute('SELECT payload FROM audit_chain ORDER BY seq DESC LIMIT 1').fetchone()
        if not row:
            raise ValueError('No signed rotation handoff found')
        handoff = json.loads(row[0])
        if (handoff.get('event') != 'AUDIT_KEY_ROTATION'
                or handoff.get('new_epoch') != new_db.name
                or handoff.get('new_anchor') != new_anchor.name
                or not old.verify_audit(handoff)):
            raise ValueError('Signed rotation handoff does not match destination')
        tip = old._ledger_tip()
        if handoff.get('old_tip') != [tip[0] - 1, old._audit_db.execute(
                'SELECT mac FROM audit_chain WHERE seq=?', (tip[0] - 1,)
            ).fetchone()[0] if tip[0] > 1 else 'GENESIS']:
            raise ValueError('Rotation handoff does not reference preceding tip')
        new = AccessGrantManager(secret_key=new_key, audit_db_path=str(new_db), audit_anchor_path=str(new_anchor))
        try:
            event = {'event': 'AUDIT_EPOCH_START', 'previous_tip': tip,
                     'previous_epoch': old_db.name, 'previous_anchor': old_anchor.name}
            event['audit_signature'] = new.sign_audit(event)
            new.append_audit(event)
            if not new.verify_audit_chain() or not old.verify_audit_chain():
                raise ValueError('Rotation recovery verification failed')
            return {'previous_tip': tip, 'new_tip': new._ledger_tip()}
        finally:
            new._audit_db.close()
    finally:
        old._audit_db.close()
