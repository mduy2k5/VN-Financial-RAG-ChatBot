from __future__ import annotations

import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "docs" / "assets" / "architecture"
SVG_PATH = ASSET_DIR / "pipeline.svg"

WIDTH = 5200
HEIGHT = 3300

INK = "#172438"
MUTED = "#607086"
BLUE = "#438fe3"
VIOLET = "#8d68d8"
TEAL = "#17a99b"
PURPLE = "#a276cf"
GRAY = "#91a0b2"
RED = "#d76f69"
GREEN = "#3d9d78"


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def text(x: float, y: float, value: str, size: int = 30, color: str = INK,
         weight: int = 400, anchor: str = "start", letter_spacing: float = 0) -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
        f'font-weight="{weight}" text-anchor="{anchor}" '
        f'letter-spacing="{letter_spacing}">{esc(value)}</text>'
    )


def multiline(x: float, y: float, lines: list[str], size: int = 25,
              color: str = MUTED, weight: int = 400, line_height: int = 34) -> str:
    spans = "".join(
        f'<tspan x="{x}" dy="{0 if index == 0 else line_height}">{esc(line)}</tspan>'
        for index, line in enumerate(lines)
    )
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
        f'font-weight="{weight}">{spans}</text>'
    )


def round_rect(x: float, y: float, w: float, h: float, fill: str,
               stroke: str = "none", radius: int = 26, stroke_width: int = 2) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def path(d: str, color: str, width: int = 7, marker: str | None = None,
         dash: str | None = None, opacity: float = 1) -> str:
    end = f' marker-end="url(#{marker})"' if marker else ""
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"'
        f'{dashed}{end}/>'
    )


def label_pill(x: int, y: int, label: str, fill: str, color: str,
               width: int | None = None, size: int = 21) -> str:
    width = width or max(150, int(len(label) * size * 0.59 + 34))
    return (
        round_rect(x, y, width, 43, fill, radius=21)
        + text(x + width / 2, y + 29, label, size, color, 700, "middle")
    )


def icon(name: str, cx: float, cy: float, size: int, color: str) -> str:
    scale = size / 100
    transform = f'translate({cx} {cy}) scale({scale}) translate(-50 -50)'
    start = (
        f'<g transform="{transform}" fill="none" stroke="{color}" stroke-width="5.5" '
        'stroke-linecap="round" stroke-linejoin="round">'
    )
    shapes: dict[str, str] = {
        "browser": (
            '<rect x="13" y="18" width="74" height="64" rx="8"/>'
            '<path d="M13 36h74M28 53h21M28 65h40"/>'
            '<circle cx="25" cy="27" r="2.5" fill="currentColor"/>'
            '<circle cx="36" cy="27" r="2.5" fill="currentColor"/>'
        ),
        "server": (
            '<rect x="19" y="17" width="62" height="66" rx="9"/>'
            '<path d="M19 39h62M19 61h62M43 28h25M43 50h25M43 72h25"/>'
            '<circle cx="31" cy="28" r="2" fill="currentColor"/>'
            '<circle cx="31" cy="50" r="2" fill="currentColor"/>'
            '<circle cx="31" cy="72" r="2" fill="currentColor"/>'
        ),
        "router": (
            '<circle cx="25" cy="50" r="9"/><circle cx="73" cy="25" r="9"/>'
            '<circle cx="73" cy="75" r="9"/><path d="M34 50h18q12 0 12-12V25M52 50q12 0 12 12v13M50 50h22"/>'
        ),
        "calculator": (
            '<rect x="23" y="13" width="54" height="74" rx="8"/>'
            '<rect x="32" y="22" width="36" height="15" rx="3"/>'
            '<path d="M34 51h5M51 51h5M68 51h1M34 68h5M51 68h5M68 68h1"/>'
            '<circle cx="40" cy="51" r="2" fill="currentColor"/>'
            '<circle cx="57" cy="68" r="2" fill="currentColor"/>'
        ),
        "book": (
            '<path d="M14 27q22-8 36 5v50q-14-13-36-5zM86 27q-22-8-36 5v50q14-13 36-5z"/>'
            '<path d="M22 41q13-3 21 2M22 54q13-3 21 2M78 41q-13-3-21 2M78 54q-13-3-21 2"/>'
        ),
        "citation": (
            '<path d="M27 13h36l15 15v59H27zM63 13v16h15M39 43h26M39 55h21"/>'
            '<path d="M44 72l8 8 17-19"/>'
        ),
        "mongo": (
            '<ellipse cx="50" cy="24" rx="31" ry="12"/><path d="M19 24v49q31 18 62 0V24"/>'
            '<path d="M19 42q31 18 62 0M19 58q31 18 62 0M50 32v35" stroke-width="3"/>'
        ),
        "search": (
            '<circle cx="43" cy="41" r="24"/><path d="M60 59l23 23M31 34h23M31 43h18M31 52h12"/>'
        ),
        "vector": (
            '<circle cx="25" cy="27" r="7"/><circle cx="69" cy="22" r="7"/>'
            '<circle cx="43" cy="53" r="7"/><circle cx="78" cy="68" r="7"/>'
            '<circle cx="22" cy="78" r="7"/><path d="M31 28l31-5M29 33l10 14M49 57l23 8M38 60l-12 12M49 49l16-21"/>'
        ),
        "cache": (
            '<path d="M20 26h59v14H20zM20 45h59v14H20zM20 64h59v14H20z"/>'
            '<circle cx="31" cy="33" r="2" fill="currentColor"/>'
            '<circle cx="31" cy="52" r="2" fill="currentColor"/>'
            '<circle cx="31" cy="71" r="2" fill="currentColor"/>'
        ),
        "model": (
            '<circle cx="50" cy="50" r="11"/><circle cx="23" cy="28" r="7"/>'
            '<circle cx="76" cy="27" r="7"/><circle cx="23" cy="73" r="7"/>'
            '<circle cx="77" cy="74" r="7"/><path d="M30 32l13 12M70 32L57 44M30 69l13-13M70 69L57 56"/>'
        ),
        "file": (
            '<path d="M27 13h34l14 15v60H27zM61 13v17h14M38 47h26M38 59h26M38 71h18"/>'
        ),
        "upload": (
            '<path d="M50 68V28M34 45l16-17 16 17M22 59v17q0 7 7 7h42q7 0 7-7V59"/>'
            '<path d="M17 49q0-10 11-11 4-18 23-18 15 0 20 14 13 1 13 13"/>'
        ),
        "queue": (
            '<path d="M18 26h64v15H18zM18 48h64v15H18zM18 70h64v15H18z"/>'
            '<path d="M28 33h26M28 55h36M28 77h20"/>'
        ),
        "worker": (
            '<circle cx="50" cy="50" r="24"/><circle cx="50" cy="50" r="8"/>'
            '<path d="M50 13v11M50 76v11M13 50h11M76 50h11M24 24l8 8M68 68l8 8M76 24l-8 8M32 68l-8 8M44 50l5 5 10-12"/>'
        ),
        "chunk": (
            '<path d="M23 14h37l16 16v56H23zM60 14v18h16M34 47h31M34 58h14M53 58h12M34 70h11M50 70h15"/>'
            '<path d="M48 43v31" stroke-dasharray="3 5"/>'
        ),
        "publish": (
            '<path d="M20 24h43l17 17v37H20zM63 24v18h17M34 62l10 10 23-27"/>'
        ),
    }
    body = shapes[name].replace("currentColor", color)
    return start + body + "</g>"


def service_card(x: int, y: int, w: int, h: int, title: str,
                 lines: list[str], icon_name: str, color: str,
                 badge: str | None = None, badge_color: str | None = None,
                 title_size: int = 36, body_size: int = 25,
                 icon_size: int = 104) -> str:
    out = [round_rect(x, y, w, h, "#ffffff", "#dbe3ed", 25, 2)]
    cx, cy = x + 76, y + h / 2
    tint = {
        BLUE: "#eef5fe", VIOLET: "#f4f0fc", TEAL: "#eaf8f5",
        PURPLE: "#f6f0fa", GREEN: "#edf8f2", RED: "#fff2f0",
        GRAY: "#f0f3f7",
    }.get(color, "#f0f4f8")
    out.append(f'<circle cx="{cx}" cy="{cy}" r="{icon_size * .54}" fill="{tint}"/>')
    out.append(icon(icon_name, cx, cy, icon_size, color))
    tx = x + 158
    line_height = body_size + 9
    text_block_height = 43 + max(0, len(lines) - 1) * line_height
    title_y = y + h / 2 - text_block_height / 2
    if badge:
        bw = max(125, int(len(badge) * 13.2 + 30))
        bx = x + w - bw - 18
        out.append(label_pill(bx, y + 15, badge, tint, badge_color or color, bw, 20))
    out.append(text(tx, title_y, title, title_size, INK, 700))
    out.append(multiline(tx, title_y + 43, lines, body_size, MUTED, 400, line_height))
    return "".join(out)


def legend_line(x: int, y: int, label: str, color: str, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="12 10"' if dashed else ""
    return (
        f'<path d="M{x} {y}h90" fill="none" stroke="{color}" stroke-width="6" '
        f'stroke-linecap="round"{dash}/>'
        + text(x + 112, y + 9, label, 26, MUTED, 500)
    )


def build_svg() -> str:
    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        '<title id="title">Kiến trúc VN Financial Understanding</title>',
        '<desc id="desc">Sơ đồ có icon, thể hiện từng bước của luồng truy vấn và luồng nạp tài liệu, cùng đường trả kết quả về giao diện.</desc>',
        '<defs>',
        '<marker id="arrow-blue" viewBox="0 0 42 42" markerWidth="42" markerHeight="42" refX="37.5" refY="21" markerUnits="userSpaceOnUse" orient="auto"><path d="M3 3L37.5 21L3 39Z" fill="#438fe3"/></marker>',
        '<marker id="arrow-violet" viewBox="0 0 42 42" markerWidth="42" markerHeight="42" refX="37.5" refY="21" markerUnits="userSpaceOnUse" orient="auto"><path d="M3 3L37.5 21L3 39Z" fill="#8d68d8"/></marker>',
        '<marker id="arrow-teal" viewBox="0 0 42 42" markerWidth="42" markerHeight="42" refX="37.5" refY="21" markerUnits="userSpaceOnUse" orient="auto"><path d="M3 3L37.5 21L3 39Z" fill="#17a99b"/></marker>',
        '<marker id="arrow-gray" markerWidth="12" markerHeight="12" refX="9" refY="6" orient="auto"><path d="M1 1l9 5-9 5z" fill="#91a0b2"/></marker>',
        '<style>text{font-family:Inter,"Segoe UI",Arial,sans-serif}</style>',
        '</defs>',
        round_rect(0, 0, WIDTH, HEIGHT, "#f4f6fa", radius=0),
    ]

    # Header and flow legend.
    out.append(text(145, 120, "KIẾN TRÚC HỆ THỐNG", 78, INK, 800, letter_spacing=1.2))
    out.append(text(150, 183, "VN FINANCIAL UNDERSTANDING  ·  LOCAL-FIRST  ·  QUERY + INGESTION", 31, MUTED, 600, letter_spacing=.8))
    out.append(text(150, 242, "Từ request đến câu trả lời; từ tài liệu đến phiên bản sẵn sàng truy vấn.", 29, MUTED, 400))
    out.append(round_rect(3290, 36, 1780, 256, "#ffffff", "#e1e7ef", 25, 2))
    out.append(text(3360, 82, "CHÚ GIẢI LUỒNG", 29, INK, 750, letter_spacing=.8))
    out.append(legend_line(3360, 132, "Nạp dữ liệu / publish", BLUE))
    out.append(legend_line(4110, 132, "Truy vấn / trả lời", VIOLET))
    out.append(legend_line(3360, 192, "Kết quả / active source", TEAL))

    # Main architecture board.
    out.append(round_rect(120, 330, 4960, 2850, "#eaf0f7", "#dde5ee", 34, 2))
    out.append(text(205, 418, "HỆ THỐNG LÕI TÀI CHÍNH", 48, INK, 800, letter_spacing=1.3))
    out.append(text(208, 468, "Các mũi tên thể hiện thứ tự xử lý và đường dữ liệu giữa các bước.", 30, MUTED, 500))

    # Query and answer lanes.
    out.append(text(220, 535, "01  ·  LUỒNG TRUY VẤN — REQUEST ĐI QUA CÁC BƯỚC NÀY", 29, "#65768b", 700, letter_spacing=1.0))
    out.append(round_rect(200, 1030, 2300, 740, "#eff7f4", "#d8e9e1", 28, 2))
    out.append(round_rect(2600, 1030, 2400, 740, "#f4f0fa", "#e3d9ef", 28, 2))

    # Arrows are drawn first so the step cards sit cleanly above them.
    # End arrow paths one unit before each target edge so the arrow tip meets the card cleanly.
    out.append(path("M960 700H1119", VIOLET, 6, "arrow-violet"))
    out.append(path("M1880 700H2229", VIOLET, 6, "arrow-violet"))
    out.append(path("M2460 850V960H1350V1029", VIOLET, 6, "arrow-violet"))
    out.append(path("M3220 850V960H3800V1029", VIOLET, 6, "arrow-violet"))
    for x in (750, 1325, 1900):
        out.append(path(f"M{x} 1440H{x + 74}", VIOLET, 6, "arrow-violet"))
    for x in (3350, 4130):
        out.append(path(f"M{x} 1440H{x + 79}", VIOLET, 6, "arrow-violet"))
    out.append(path("M2225 1660V1859", VIOLET, 6, "arrow-violet"))
    out.append(path("M4560 1660V1825H3000V1859", VIOLET, 6, "arrow-violet"))
    out.append(path("M1750 1980H150V700H259", TEAL, 6, "arrow-teal"))

    out.append(service_card(260, 575, 700, 250, "Web UI", ["Nginx · :8093", "người dùng gửi câu hỏi"], "browser", BLUE,
                            title_size=38, body_size=28, icon_size=100))
    out.append(service_card(1120, 575, 760, 250, "FastAPI", ["REST · :8092", "nhận query và điều phối"], "server", BLUE,
                            title_size=40, body_size=28, icon_size=100))
    out.append(service_card(2230, 555, 1220, 295, "Chọn notebook + source scope", ["xác nhận revision notebook", "ghim nguồn / version được phép đọc", "route theo yêu cầu: phân tích số hoặc đọc nguồn"], "router", TEAL,
                            title_size=38, body_size=27, icon_size=104))

    out.append(text(250, 1110, "A  ·  PHÂN TÍCH TÀI CHÍNH", 27, "#3c806c", 800, letter_spacing=.8))
    out.append(text(2650, 1110, "B  ·  ĐỌC NỘI DUNG NGUỒN", 27, "#715893", 800, letter_spacing=.8))
    finance_steps = [
        (250, "Facts từ MongoDB", ["đọc trong source scope", "normalized values theo kỳ"], "mongo", TEAL),
        (825, "Decimal executor", ["tính tổng · tỷ lệ · so sánh", "LLM không tính số"], "calculator", VIOLET),
        (1400, "Diễn giải", ["Ollama khi cần narrative", "không thay phép tính"], "model", PURPLE),
        (1975, "Kiểm định kết quả", ["kiểm tra kỳ · đơn vị · citation", "gắn trạng thái câu trả lời"], "citation", GREEN),
    ]
    for x, title, lines, icon_name, color in finance_steps:
        badge = "TÙY CHỌN" if title == "Diễn giải" else None
        out.append(service_card(x, 1220, 500, 440, title, lines, icon_name, color,
                                badge=badge, badge_color="#78559d", title_size=29, body_size=22, icon_size=84))

    reader_steps = [
        (2650, "Tìm bằng chứng", ["BM25 trong source scope", "Qdrant dense nếu bật"], "search", VIOLET),
        (3430, "Mở block nguồn", ["đọc section / table gốc", "MongoDB + version đã ghim"], "book", TEAL),
        (4210, "Notebook Reader", ["trích xuất nội dung nguồn", "citation + source span"], "citation", GREEN),
    ]
    for x, title, lines, icon_name, color in reader_steps:
        out.append(service_card(x, 1220, 700, 440, title, lines, icon_name, color,
                                title_size=31, body_size=23, icon_size=88))

    out.append(label_pill(180, 875, "TRẢ KẾT QUẢ VỀ UI", "#e7f6f2", "#247e73", 350, 20))
    out.append(label_pill(3900, 875, "ACTIVE SOURCE → QUERY TIẾP THEO", "#e7f6f2", "#247e73", 520, 19))
    out.append(service_card(1750, 1860, 1700, 240, "Ghi answer run + trả kết quả", ["lưu snapshot nguồn và citations vào MongoDB", "response quay lại UI: answer · trạng thái · citations"], "citation", GREEN,
                            title_size=36, body_size=25, icon_size=92))

    # Ingestion is a second, independent runtime flow that publishes data for later queries.
    out.append(text(220, 2240, "02  ·  LUỒNG NẠP DỮ LIỆU — PUBLISH XONG THÌ QUERY MỚI ĐỌC VERSION MỚI", 29, "#65768b", 700, letter_spacing=.9))
    ingestion_x = [200, 900, 1600, 2300, 3000, 3700, 4400]
    for index in range(len(ingestion_x) - 1):
        out.append(path(f"M{ingestion_x[index] + 620} 2600H{ingestion_x[index + 1] - 1}", BLUE, 6, "arrow-blue"))
    # Successful publication makes the new active source available to future requests.
    out.append(path("M5020 2600H5050V700H3451", TEAL, 6, "arrow-teal"))

    ingestion_steps = [
        (200, "TXT / Markdown", ["tài liệu đầu vào", "giữ nội dung + page/span"], "file"),
        (900, "Upload API", ["nhận và kiểm tra request", "tạo bản ghi nguồn"], "upload"),
        (1600, "MongoDB job queue", ["ghi job + trạng thái", "ghim revision nguồn"], "queue"),
        (2300, "Ingestion worker", ["lấy job bất đồng bộ", "heartbeat · retry khi lỗi"], "worker"),
        (3000, "Parse + normalize", ["tách block / table / chunk", "giữ liên kết về nguồn gốc"], "chunk"),
        (3700, "Build retrieval index", ["tạo BM25 index", "Qdrant khi dense bật"], "search"),
        (4400, "CAS → MongoDB", ["index sẵn sàng + revision đúng", "đổi active projection", "lỗi thì giữ version cũ"], "publish"),
    ]
    for x, title, lines, icon_name in ingestion_steps:
        out.append(service_card(x, 2370, 620, 460, title, lines, icon_name, BLUE,
                                title_size=30, body_size=22, icon_size=84))

    out.append(round_rect(190, 2950, 4820, 165, "#f7fafc", "#dce5ed", 22, 2))
    out.append(text(235, 3012, "KẾT QUẢ CỦA LUỒNG NẠP", 25, INK, 800, letter_spacing=.8))
    out.append(text(235, 3070, "Thành công: query kế tiếp đọc phiên bản mới.", 24, "#34795d", 600))
    out.append(text(2710, 3070, "Thất bại: phiên bản đang hoạt động vẫn được giữ.", 24, "#8c6258", 600))

    out.append('</svg>')
    return "\n".join(out)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(build_svg(), encoding="utf-8")
    print(SVG_PATH)


if __name__ == "__main__":
    main()
