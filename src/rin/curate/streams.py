"""Which frames belong to one stream: one camera, on one day, in one split.

Shared by every curation step that treats a camera's footage as a whole (centre, dedupe, select),
so they can't disagree. A stream never mixes splits (a camera's val clips mustn't continue its
pool stream, D-13) or days (yesterday's last frame isn't "the frame before" today's first).
"""

from __future__ import annotations


def stream_sql(media: str = "d") -> str:
    """A SQL expression naming the stream of a row of the `media` table aliased `media`.

    With a camera: 'camera <camera> (<site>, <split>, <day>)', where site and day read 'no site'
    and 'no day' when the media doesn't know them. Without a camera: 'media <media_id>', so each
    such file is its own stream.
    """
    d = media
    return (
        f"CASE WHEN {d}.camera IS NULL THEN 'media ' || {d}.media_id "
        f"ELSE 'camera ' || {d}.camera || ' (' || coalesce({d}.site, 'no site') || ', ' || {d}.split"
        f" || ', ' || coalesce(strftime(CAST({d}.started_at AS DATE), '%Y-%m-%d'), 'no day') || ')' END"
    )
