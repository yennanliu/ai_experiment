"""Exercise 3 — the token binds the arguments and cannot bind the world.

    Replace the in-memory note map with a temporary SQLite database. Re-check
    authorization and containment inside the mutation transaction.

Reading of the exercise: "re-check" implies the first check could go stale, so
the solution's job is to make it go stale and watch. Two things are true when
round one issues the elicitation and can be false when round two arrives --
the note is inside the workspace, and the workspace is authorized -- so both
are changed between the rounds. Putting the checks inside the transaction is
then not defensive style; it is the only place they are still true when the
row is removed.

**ANSWER: the notes live in a temporary database and the delete re-checks both
facts in one transaction.** The MRTR flow completes, the table goes **5** rows
to **4**, and the deleted id is the one the user confirmed.

**FINDING: containment can lapse between rounds, and `candidateIds` does not
notice.** Moving a confirmed candidate's URI to `file:///tmp` after round one
leaves it in the sealed `candidateIds`, and the delete is refused with
`selected note is outside workspace`. The token records which notes were
eligible, which is a fact about the past.

**FINDING: authorization can lapse too, and the token has no field for it.**
Revoking the workspace between rounds refuses the retry with `workspace is not
authorized`. `requestState` binds principal, method and an arguments digest --
all properties of the request -- and grants are a property of the server, so
no sealed token could have carried this.

**FINDING: the transaction is what makes a refused check harmless.** Each
refusal leaves the table at **5** rows, because the `SELECT`, both checks and
the `DELETE` share one `with sqlite3.connect(...)` block. Checking after a
committed delete would be a check on a row that no longer exists.

Structure: `SqliteNotes` is the store plus `delete_if_allowed`, the one method
that holds the transaction; `SqliteNotesServer` swaps it into the lesson's own
`NotesServer` by overriding the single method that mutates.
"""

from __future__ import annotations

import pathlib
import sqlite3
import tempfile

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "12-mcp-roots-and-elicitation"
WORKSPACE = "file:///Users/alice/Documents/Notes"
OUTSIDE = "file:///tmp/moved.md"


class SqliteNotes:
    """The note map on disk, with the delete guarded inside its transaction."""

    def __init__(self, ref, path, rows):
        self.ref, self.path = ref, path
        with self._connect() as db:
            db.execute("CREATE TABLE notes (id TEXT PRIMARY KEY, title TEXT, uri TEXT)")
            db.executemany("INSERT INTO notes VALUES (?, ?, ?)",
                           [(i, n["title"], n["uri"]) for i, n in rows.items()])

    def _connect(self):
        return sqlite3.connect(self.path, isolation_level="IMMEDIATE", timeout=0.5)

    def items(self):
        with self._connect() as db:
            return [(i, {"title": t, "uri": u})
                    for i, t, u in db.execute("SELECT id, title, uri FROM notes")]

    def move(self, note_id, uri):
        with self._connect() as db:
            db.execute("UPDATE notes SET uri = ? WHERE id = ?", (uri, note_id))

    def __len__(self):
        return len(self.items())

    def delete_if_allowed(self, note_id, workspace_uri, authorized):
        """Both re-checks and the mutation, in one transaction or none of it."""
        with self._connect() as db:
            row = db.execute("SELECT uri FROM notes WHERE id = ?", (note_id,)).fetchone()
            if row is None:
                raise self.ref.McpError(-32602, "selected note no longer exists")
            if workspace_uri not in authorized:
                raise self.ref.McpError(-32602, "workspace is not authorized")
            if not self.ref.uri_within_workspace(workspace_uri, row[0]):
                raise self.ref.McpError(-32602, "selected note is outside workspace")
            db.execute("DELETE FROM notes WHERE id = ?", (note_id,))


def make_server(ref, notes):
    """The lesson's NotesServer with its one mutating method redirected."""

    class SqliteNotesServer(ref.NotesServer):
        def _delete_note_once(self, *, nonce, expires_at, note_id, workspace_uri):
            self.replay_store.claim_and_consume(
                nonce, expires_at=expires_at,
                operation=lambda: notes.delete_if_allowed(
                    note_id, workspace_uri, self.authorized_workspaces))

    server = SqliteNotesServer()
    server.notes = notes
    return server


def run(ref, server, before=None):
    """One full MRTR round trip, with `before` applied between the rounds."""
    result = server.dispatch(ref.tool_request(1))["result"]
    schema = result["inputRequests"]["delete_choice"]["params"]["requestedSchema"]
    chosen = schema["properties"]["note_id"]["enum"][-1]
    if before is not None:
        before(chosen)
    retry = ref.tool_request(2)
    retry["params"].update({"requestState": result["requestState"], "inputResponses": {
        "delete_choice": {"action": "accept",
                          "content": {"note_id": chosen, "confirm": True}}}})
    response = server.dispatch(retry)
    if "error" in response:
        return chosen, response["error"]["message"]
    return chosen, response["result"]["structuredContent"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as directory:
        def store(name):
            return SqliteNotes(ref, str(pathlib.Path(directory) / name), ref.DEFAULT_NOTES)

        happy, moved, revoked = store("a.sqlite3"), store("b.sqlite3"), store("c.sqlite3")
        before = len(happy)
        chosen, deleted = run(ref, make_server(ref, happy))
        _, move_refusal = run(ref, make_server(ref, moved),
                              before=lambda nid: moved.move(nid, OUTSIDE))
        revoked_server = make_server(ref, revoked)
        _, revoke_refusal = run(ref, revoked_server,
                                before=lambda _: revoked_server.authorized_workspaces.clear())
        return {
            "before": before, "after": len(happy), "chosen": chosen, "deleted": deleted,
            "move_refusal": move_refusal, "moved_rows": len(moved),
            "revoke_refusal": revoke_refusal, "revoked_rows": len(revoked),
            "state_fields": ["argumentsDigest", "candidateIds", "expiresAt", "method",
                             "nonce", "phase", "principal"],
        }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the notes live in a temporary database and the flow deletes one row",
            all([result["before"] == 5, result["after"] == 4,
                 result["deleted"] == {"deleted": True, "noteId": result["chosen"]}]),
            f"the table goes {result['before']} rows to {result['after']} and the result is "
            f"{result['deleted']}. NotesServer is unchanged but for the one mutating method",
        ),
        practice.Check(
            "FINDING: containment can lapse between rounds, and candidateIds does not notice",
            all([result["move_refusal"] == "selected note is outside workspace",
                 result["moved_rows"] == 5]),
            f"moving a confirmed candidate's URI to {OUTSIDE} after round one leaves it in "
            f"the sealed candidateIds and the delete is refused, {result['move_refusal']!r}, "
            f"with {result['moved_rows']} rows present. The token records which notes were "
            "eligible, which is a fact about the past",
        ),
        practice.Check(
            "FINDING: authorization can lapse too, and the token has no field for it",
            all([result["revoke_refusal"] == "workspace is not authorized",
                 result["revoked_rows"] == 5,
                 "authorizedWorkspaces" not in result["state_fields"]]),
            f"revoking the workspace between rounds refuses the retry, "
            f"{result['revoke_refusal']!r}. requestState binds {result['state_fields']} -- "
            "all properties of the request -- and grants are a property of the server, so no "
            "sealed token could have carried this",
        ),
        practice.Check(
            "FINDING: the transaction is what makes a refused check harmless",
            all([result["moved_rows"] == result["before"],
                 result["revoked_rows"] == result["before"]]),
            f"each refusal leaves the table at {result['before']} rows, because the SELECT, "
            "both checks and the DELETE share one connection block. Checking after a "
            "committed delete would be a check on a row that no longer exists",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
