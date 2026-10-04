from __future__ import annotations

import ast
import builtins
import io
import keyword
import textwrap
import token
import tokenize
from dataclasses import dataclass

from swe_memory_policy.image_response_rencent3text.tools.model import (
    HARD_NEWLINE_MARKER,
    CompactRegion,
    StyledAtom,
)

BLOCK_WORDS = {
    "class",
    "def",
    "if",
    "elif",
    "else",
    "for",
    "while",
    "try",
    "except",
    "finally",
    "with",
    "match",
    "case",
}
BUILTIN_NAMES = frozenset(dir(builtins))


@dataclass(frozen=True)
class PythonCompaction:
    atoms: list[StyledAtom]
    open_braces: int
    close_braces: int
    tokenize_complete: bool


@dataclass(frozen=True)
class PythonCommentRemoval:
    text: str
    comment_count: int
    comment_characters: int
    comment_lines: int


@dataclass(frozen=True)
class PythonVisualRemoval:
    text: str
    comment_count: int
    comment_characters: int
    comment_lines: int
    docstring_count: int
    docstring_characters: int
    docstring_lines: int
    docstring_ast_complete: bool


def _tokens_best_effort(
    source: str,
) -> tuple[list[tokenize.TokenInfo], bool]:
    """Return tokens emitted before an incomplete fragment stops tokenization."""

    items: list[tokenize.TokenInfo] = []
    try:
        for item in tokenize.generate_tokens(io.StringIO(source).readline):
            items.append(item)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return items, False
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for item in items:
        if item.type != token.OP:
            continue
        if item.string in "([{":
            stack.append(item.string)
        elif item.string in pairs and (not stack or stack.pop() != pairs[item.string]):
            return items, False
    if stack:
        return items, False
    return items, True


def strip_python_comments(source: str) -> PythonCommentRemoval:
    """Remove only lexical Python comments while keeping strings untouched.

    Tokenization is consumed incrementally so comments emitted before the
    expected EOF error of a partial ``sed``/``grep`` fragment are still removed.
    The caller retains the untouched observation separately for auditability.
    """

    tokens, _ = _tokens_best_effort(source)
    comments = [item for item in tokens if item.type == token.COMMENT]
    if not comments:
        return PythonCommentRemoval(source, 0, 0, 0)

    by_row: dict[int, list[tokenize.TokenInfo]] = {}
    for item in comments:
        by_row.setdefault(item.start[0], []).append(item)

    lines = source.splitlines(keepends=True)
    for row, row_comments in by_row.items():
        if row < 1 or row > len(lines):
            continue
        line = lines[row - 1]
        for item in sorted(
            row_comments, key=lambda value: value.start[1], reverse=True
        ):
            start = min(item.start[1], len(line))
            end = min(item.end[1], len(line))
            # Whitespace immediately before a removed inline comment has no
            # remaining visual meaning and otherwise creates an empty gap.
            prefix = line[:start].rstrip(" \t")
            line = prefix + line[end:]
        lines[row - 1] = line

    return PythonCommentRemoval(
        "".join(lines),
        len(comments),
        sum(len(item.string) for item in comments),
        len(by_row),
    )


def _docstring_expressions(tree: ast.AST) -> list[ast.Expr]:
    """Return only semantic module/class/function docstring statements."""

    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    expressions: list[ast.Expr] = []
    for owner in ast.walk(tree):
        if not isinstance(owner, owners):
            continue
        body = getattr(owner, "body", ())
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            expressions.append(first)
    return sorted(
        expressions,
        key=lambda item: (item.lineno, item.col_offset),
        reverse=True,
    )


def _character_column(line: str, byte_column: int) -> int:
    """Convert CPython AST's UTF-8 byte column to a string character column."""

    encoded = line.encode("utf-8")
    return len(encoded[:byte_column].decode("utf-8", errors="ignore"))


def _remove_docstring_expression(lines: list[str], expression: ast.Expr) -> None:
    """Remove one AST-proven docstring without disturbing adjacent code."""

    start_row = expression.lineno - 1
    end_row = getattr(expression, "end_lineno", expression.lineno) - 1
    if start_row < 0 or end_row >= len(lines):
        return
    start_col = _character_column(lines[start_row], expression.col_offset)
    end_col = _character_column(
        lines[end_row], getattr(expression, "end_col_offset", expression.col_offset)
    )
    prefix = lines[start_row][:start_col]
    suffix = lines[end_row][end_col:]

    # Inline suites can legally place the docstring before another statement:
    # ``def f(): "doc"; return 1``.  Remove the separator with the docstring so
    # the remaining visual code is still compactable as Python.
    stripped_suffix = suffix.lstrip(" \t")
    if stripped_suffix.startswith(";"):
        suffix = stripped_suffix[1:].lstrip(" \t")
        if prefix and not prefix[-1].isspace():
            prefix += " "

    ending = ""
    if suffix.endswith("\r\n"):
        ending = "\r\n"
    elif suffix.endswith(("\n", "\r")):
        ending = suffix[-1]

    replacement = prefix + suffix
    if not replacement.rstrip("\r\n").strip(" \t"):
        replacement = ending
    lines[start_row : end_row + 1] = [replacement]


def strip_python_nonsemantic_text(source: str) -> PythonVisualRemoval:
    """Hide comments and AST-proven docstrings from the visual rendering.

    Ordinary strings, including assigned/returned triple-quoted strings, are
    retained.  If the fragment cannot be parsed as a complete Python module,
    docstring removal fails closed and only lexical ``#`` comments are removed.
    The untouched observation remains available in the request audit artifacts.
    """

    comment_removal = strip_python_comments(source)
    try:
        tree = ast.parse(comment_removal.text)
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return PythonVisualRemoval(
            text=comment_removal.text,
            comment_count=comment_removal.comment_count,
            comment_characters=comment_removal.comment_characters,
            comment_lines=comment_removal.comment_lines,
            docstring_count=0,
            docstring_characters=0,
            docstring_lines=0,
            docstring_ast_complete=False,
        )

    expressions = _docstring_expressions(tree)
    if not expressions:
        return PythonVisualRemoval(
            text=comment_removal.text,
            comment_count=comment_removal.comment_count,
            comment_characters=comment_removal.comment_characters,
            comment_lines=comment_removal.comment_lines,
            docstring_count=0,
            docstring_characters=0,
            docstring_lines=0,
            docstring_ast_complete=True,
        )

    original_lines = comment_removal.text.splitlines(keepends=True)
    removed_characters = 0
    removed_rows: set[int] = set()
    for expression in expressions:
        segment = ast.get_source_segment(comment_removal.text, expression)
        if segment is not None:
            removed_characters += len(segment)
        removed_rows.update(
            range(
                expression.lineno,
                getattr(expression, "end_lineno", expression.lineno) + 1,
            )
        )
        _remove_docstring_expression(original_lines, expression)

    return PythonVisualRemoval(
        text="".join(original_lines),
        comment_count=comment_removal.comment_count,
        comment_characters=comment_removal.comment_characters,
        comment_lines=comment_removal.comment_lines,
        docstring_count=len(expressions),
        docstring_characters=removed_characters,
        docstring_lines=len(removed_rows),
        docstring_ast_complete=True,
    )


def _token_style(
    item: tokenize.TokenInfo,
    *,
    definition_positions: set[tuple[int, int]],
    decorator_rows: set[int],
) -> str:
    if item.type == token.COMMENT:
        return "comment"
    if item.type == token.STRING:
        return "string"
    if item.type == token.NUMBER:
        return "number"
    if item.type == token.OP:
        return "decorator" if item.start[0] in decorator_rows else "operator"
    if item.type != token.NAME:
        return "plain"
    if item.start in definition_positions:
        return "definition"
    if keyword.iskeyword(item.string) or item.string in {"match", "case"}:
        return "keyword"
    if item.start[0] in decorator_rows:
        return "decorator"
    if item.string in BUILTIN_NAMES:
        return "builtin"
    return "plain"


def _style_grid(lines: list[str], tokens: list[tokenize.TokenInfo]) -> list[list[str]]:
    styles = [["plain"] * len(line.rstrip("\r\n")) for line in lines]
    meaningful = [
        item
        for item in tokens
        if item.type
        not in {
            token.ENCODING,
            token.INDENT,
            token.DEDENT,
            token.NEWLINE,
            tokenize.NL,
            token.ENDMARKER,
        }
    ]
    definition_positions: set[tuple[int, int]] = set()
    decorator_rows: set[int] = set()
    expect_definition = False
    for item in meaningful:
        if item.type == token.OP and item.string == "@":
            decorator_rows.add(item.start[0])
        if item.type == token.NAME and item.string in {"class", "def"}:
            expect_definition = True
            continue
        if expect_definition and item.type == token.NAME:
            definition_positions.add(item.start)
            expect_definition = False
        elif item.type not in {token.COMMENT}:
            expect_definition = False

    for item in meaningful:
        style = _token_style(
            item,
            definition_positions=definition_positions,
            decorator_rows=decorator_rows,
        )
        start_row, start_col = item.start
        end_row, end_col = item.end
        for row in range(start_row, end_row + 1):
            if row < 1 or row > len(styles):
                continue
            left = start_col if row == start_row else 0
            right = end_col if row == end_row else len(styles[row - 1])
            left = max(0, min(left, len(styles[row - 1])))
            right = max(left, min(right, len(styles[row - 1])))
            for column in range(left, right):
                styles[row - 1][column] = style
    return styles


def _header_colon(
    statement: list[tokenize.TokenInfo],
) -> tokenize.TokenInfo | None:
    significant = [
        item
        for item in statement
        if item.type not in {tokenize.NL, token.NEWLINE, token.COMMENT}
    ]
    if not significant:
        return None
    names = [item.string for item in significant if item.type == token.NAME]
    first = names[0] if names else ""
    if first == "async" and len(names) > 1:
        first = names[1]
    if first not in BLOCK_WORDS:
        return None
    depth = 0
    colon: tokenize.TokenInfo | None = None
    for item in significant:
        if item.type != token.OP:
            continue
        if item.string in "([{":
            depth += 1
        elif item.string in ")]}":
            depth = max(0, depth - 1)
        elif item.string == ":" and depth == 0:
            colon = item
    return colon


def compact_python_text(source: str, *, complete: bool) -> PythonCompaction:
    """Encode physical lines as flow text and suites as visible braces."""

    source = textwrap.dedent(source)
    source = strip_python_nonsemantic_text(source).text
    lines = source.splitlines(keepends=True)
    if not lines:
        return PythonCompaction([], 0, 0, True)
    tokens, tokenize_complete = _tokens_best_effort(source)
    if not tokenize_complete:
        atoms: list[StyledAtom] = []
        for line in lines:
            raw = line.rstrip("\r\n")
            # A partial sed/cat excerpt can fail tokenization even though it is
            # recognizably Python.  Newlines are already represented explicitly,
            # so retaining physical indentation only creates large visual gaps.
            display = raw.lstrip(" \t")
            if not display:
                continue
            atoms.append(StyledAtom(display, "plain"))
            if len(line) != len(raw):
                atoms.append(
                    StyledAtom(
                        HARD_NEWLINE_MARKER,
                        "structure",
                        generated=True,
                        generated_kind="newline",
                    )
                )
        return PythonCompaction(atoms, 0, 0, False)

    styles = _style_grid(lines, tokens)
    opens_after: dict[tuple[int, int], int] = {}
    closes_before: dict[int, int] = {}
    inline_closes: dict[int, int] = {}
    generated_indent_stack: list[bool] = []
    statement: list[tokenize.TokenInfo] = []
    last_statement: list[tokenize.TokenInfo] = []
    for item in tokens:
        if item.type == token.INDENT:
            colon = _header_colon(last_statement)
            generated_indent_stack.append(colon is not None)
            if colon is not None:
                opens_after[colon.end] = opens_after.get(colon.end, 0) + 1
            continue
        if item.type == token.DEDENT:
            if not generated_indent_stack.pop():
                continue
            row = item.start[0]
            # generate_tokens emits implicit DEDENTs on the synthetic EOF row.
            if complete or bool(item.line):
                closes_before[row] = closes_before.get(row, 0) + 1
            continue
        if item.type == token.NEWLINE:
            colon = _header_colon(statement)
            if colon is not None:
                significant_after = [
                    part
                    for part in statement
                    if part.start >= colon.end
                    and part.type not in {token.COMMENT, tokenize.NL, token.NEWLINE}
                ]
                if significant_after:
                    opens_after[colon.end] = opens_after.get(colon.end, 0) + 1
                    inline_closes[item.start[0]] = (
                        inline_closes.get(item.start[0], 0) + 1
                    )
            last_statement = statement
            statement = []
            continue
        if item.type not in {
            token.ENCODING,
            token.ENDMARKER,
            token.INDENT,
            token.DEDENT,
        }:
            statement.append(item)
    atoms: list[StyledAtom] = []
    open_count = 0
    close_count = 0

    def append_runs(text: str, row_styles: list[str], offset: int) -> None:
        cursor = 0
        while cursor < len(text):
            absolute = offset + cursor
            count = opens_after.get((row_number, absolute), 0)
            if count:
                atoms.append(
                    StyledAtom(
                        " " + " ".join("{" for _ in range(count)),
                        "structure",
                        generated=True,
                        generated_kind="open_brace",
                    )
                )
            style = row_styles[absolute] if absolute < len(row_styles) else "plain"
            end = cursor + 1
            while end < len(text):
                next_absolute = offset + end
                if opens_after.get((row_number, next_absolute), 0):
                    break
                next_style = (
                    row_styles[next_absolute]
                    if next_absolute < len(row_styles)
                    else "plain"
                )
                if next_style != style:
                    break
                end += 1
            atoms.append(StyledAtom(text[cursor:end], style))
            cursor = end
        count = opens_after.get((row_number, offset + len(text)), 0)
        if count:
            atoms.append(
                StyledAtom(
                    " " + " ".join("{" for _ in range(count)),
                    "structure",
                    generated=True,
                    generated_kind="open_brace",
                )
            )

    for row_number, line in enumerate(lines, start=1):
        raw = line.rstrip("\r\n")
        if not raw.strip(" \t"):
            # Removed comment-only and existing blank rows should not leave an
            # empty newline marker in the compact visual transcript.
            continue
        leading = len(raw) - len(raw.lstrip(" \t"))
        before = closes_before.get(row_number, 0)
        if before:
            atoms.append(
                StyledAtom(
                    " ".join("}" for _ in range(before)) + " ",
                    "structure",
                    generated=True,
                    generated_kind="close_brace",
                )
            )
            close_count += before
        before_open_count = sum(
            count for (row, _), count in opens_after.items() if row == row_number
        )
        append_runs(raw[leading:], styles[row_number - 1], leading)
        open_count += before_open_count
        inline = inline_closes.get(row_number, 0)
        if inline:
            atoms.append(
                StyledAtom(
                    " " + " ".join("}" for _ in range(inline)),
                    "structure",
                    generated=True,
                    generated_kind="close_brace",
                )
            )
            close_count += inline
        if len(line) != len(raw):
            atoms.append(
                StyledAtom(
                    HARD_NEWLINE_MARKER,
                    "structure",
                    generated=True,
                    generated_kind="newline",
                )
            )

    eof_closes = closes_before.get(len(lines) + 1, 0)
    if eof_closes:
        atoms.append(
            StyledAtom(
                " ".join("}" for _ in range(eof_closes)),
                "structure",
                generated=True,
                generated_kind="close_brace",
            )
        )
        close_count += eof_closes
    return PythonCompaction(atoms, open_count, close_count, True)


def _decorated_start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", ())
    return min([getattr(node, "lineno", 1), *(item.lineno for item in decorators)])


def build_python_regions(source: str, *, complete: bool) -> list[CompactRegion]:
    """Split a complete module into Prelude, class, method, and function regions."""

    lines = source.splitlines(keepends=True)
    try:
        tree = ast.parse(source) if complete else None
    except (SyntaxError, ValueError):
        tree = None
    if tree is None:
        compacted = compact_python_text(source, complete=False)
        return [
            CompactRegion(
                kind="python_fragment",
                title="Python fragment",
                atoms=compacted.atoms,
                source_start_line=1,
                source_end_line=len(lines),
            )
        ]

    regions: list[CompactRegion] = []
    cursor = 1

    def add_region(
        kind: str,
        title: str,
        start: int,
        end: int,
        *,
        opens: int = 0,
        closes: int = 0,
        region_complete: bool = True,
    ) -> None:
        if end < start:
            return
        text = "".join(lines[start - 1 : end])
        compacted = compact_python_text(text, complete=region_complete)
        atoms = list(compacted.atoms)
        if not atoms:
            # A region containing only comments/docstrings should disappear
            # completely rather than leaving an empty gray visual container.
            return
        missing_opens = max(0, opens - compacted.open_braces)
        if missing_opens:
            insertion = len(atoms)
            if atoms and atoms[-1].generated_kind == "newline":
                insertion -= 1
            atoms.insert(
                insertion,
                StyledAtom(
                    " " + " ".join("{" for _ in range(missing_opens)),
                    "structure",
                    generated=True,
                    generated_kind="open_brace",
                ),
            )
        local_closes = (
            max(
                0,
                compacted.open_braces - compacted.close_braces - opens,
            )
            if not region_complete
            else 0
        )
        if local_closes:
            atoms.append(
                StyledAtom(
                    " " + " ".join("}" for _ in range(local_closes)),
                    "structure",
                    generated=True,
                    generated_kind="close_brace",
                )
            )
        if closes:
            atoms.append(
                StyledAtom(
                    " " + " ".join("}" for _ in range(closes)),
                    "structure",
                    generated=True,
                    generated_kind="close_brace",
                )
            )
        regions.append(
            CompactRegion(
                kind=kind,
                title=title,
                atoms=atoms,
                source_start_line=start,
                source_end_line=end,
            )
        )

    structural = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    for node in structural:
        start = _decorated_start(node)
        if cursor < start:
            title = "Prelude" if not regions else "Module"
            add_region("prelude" if not regions else "module", title, cursor, start - 1)
        end = getattr(node, "end_lineno", start)
        if isinstance(node, ast.ClassDef):
            methods = [
                child
                for child in node.body
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
            ]
            if not methods:
                add_region("class", f"class {node.name}", start, end)
            else:
                method_starts = [_decorated_start(method) for method in methods]
                add_region(
                    "class",
                    f"class {node.name}",
                    start,
                    method_starts[0] - 1,
                    opens=1,
                    region_complete=False,
                )
                for index, method in enumerate(methods):
                    method_start = method_starts[index]
                    method_end = (
                        method_starts[index + 1] - 1
                        if index + 1 < len(methods)
                        else end
                    )
                    add_region(
                        "method",
                        f"{node.name}.{method.name}",
                        method_start,
                        method_end,
                        closes=1 if index + 1 == len(methods) else 0,
                    )
        else:
            add_region("function", f"def {node.name}", start, end)
        cursor = end + 1
    if cursor <= len(lines):
        title = "Prelude" if not regions else "Module"
        add_region("prelude" if not regions else "module", title, cursor, len(lines))
    return regions


def brace_counts(regions: list[CompactRegion]) -> tuple[int, int]:
    opens = 0
    closes = 0
    for region in regions:
        for atom in region.atoms:
            if atom.generated_kind == "open_brace":
                opens += atom.text.count("{")
            elif atom.generated_kind == "close_brace":
                closes += atom.text.count("}")
    return opens, closes
