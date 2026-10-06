from dataclasses import dataclass, field
from pathlib import Path
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from ..schemas import StatementType


@dataclass
class MetricDefinition:
    canonical_name: str
    code: Optional[str]
    statement_type: StatementType
    statement_name_vi: str
    section: str
    aliases: List[str] = field(default_factory=list)
    metric_id: str = ""


STATEMENT_NAMES_VI: Dict[StatementType, str] = {
    StatementType.BALANCE_SHEET: "Bảng cân đối kế toán",
    StatementType.INCOME_STATEMENT: "Báo cáo kết quả hoạt động kinh doanh",
    StatementType.CASH_FLOW: "Báo cáo lưu chuyển tiền tệ",
    StatementType.NOTES: "Thuyết minh báo cáo tài chính",
    StatementType.AUDIT_OPINION: "Báo cáo kiểm toán",
    StatementType.ACCOUNTING_POLICY: "Chính sách kế toán",
    StatementType.GENERAL: "Tổng quan chung",
}

# Ánh xạ từ các từ khóa nhận diện phần báo cáo tài chính
STATEMENT_KEYWORD_ROUTING: Dict[StatementType, List[str]] = {
    StatementType.BALANCE_SHEET: [
        "bảng cân đối kế toán",
        "báo cáo tình hình tài chính",
        "bcđkt",
        "tình hình tài chính",
        "cân đối kế toán",
        "b01-dn",
        "b01a-dn",
        "b02/tctd",
    ],
    StatementType.INCOME_STATEMENT: [
        "báo cáo kết quả hoạt động kinh doanh",
        "kết quả hoạt động kinh doanh",
        "báo cáo kết quả kinh doanh",
        "kqkd",
        "kết quả kinh doanh",
        "b02-dn",
        "b03/tctd",
    ],
    StatementType.CASH_FLOW: [
        "báo cáo lưu chuyển tiền tệ",
        "lưu chuyển tiền tệ",
        "lưu chuyển tiền",
        "lctt",
        "b03-dn",
        "b04/tctd",
    ],
    StatementType.NOTES: [
        "thuyết minh báo cáo tài chính",
        "thuyết minh bctc",
        "thuyết minh",
        "b09-dn",
        "b05/tctd",
    ],
}


class FinancialMetricMapper:
    """
    Bộ ánh xạ chuyên sâu các chỉ tiêu tài chính chuẩn hóa (Canonical Financial Metric Ontology)
    cho 4 phần cốt lõi của Báo cáo tài chính Việt Nam:
    1. Bảng cân đối kế toán (Báo cáo tình hình tài chính)
    2. Báo cáo kết quả hoạt động kinh doanh
    3. Báo cáo lưu chuyển tiền tệ
    4. Thuyết minh báo cáo tài chính
    """

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = (
                Path(__file__).resolve().parent.parent.parent.parent
                / "configs"
                / "financial_terms_vi.json"
            )
        self.config_path = config_path
        self.metrics_list: List[MetricDefinition] = []
        self._alias_to_metric: Dict[str, MetricDefinition] = {}
        self._code_to_metric: Dict[Tuple[StatementType, str], MetricDefinition] = {}
        self._load_taxonomy()

    def _load_taxonomy(self):
        """Khởi tạo toàn bộ danh mục chỉ tiêu chuẩn hóa theo Thông tư 200 & TT 49 Ngân hàng."""
        # 1. BẢNG CÂN ĐỐI KẾ TOÁN (BÁO CÁO TÌNH HÌNH TÀI CHÍNH)
        bs = StatementType.BALANCE_SHEET
        bs_name = STATEMENT_NAMES_VI[bs]

        self._register(
            MetricDefinition(
                canonical_name="TỔNG CỘNG TÀI SẢN",
                code="270",
                statement_type=bs,
                statement_name_vi=bs_name,
                section="tai_san",
                aliases=[
                    "tổng cộng tài sản", "tổng tài sản", "quy mô tài sản", "tổng ts",
                    "tài sản tổng cộng", "total assets",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="TÀI SẢN NGẮN HẠN",
                code="100",
                statement_type=bs,
                statement_name_vi=bs_name,
                section="tai_san_ngan_han",
                aliases=[
                    "tài sản ngắn hạn", "tổng tài sản ngắn hạn", "tsnh", "current assets",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="TIỀN VÀ TƯƠNG ĐƯƠNG TIỀN",
                code="110",
                statement_type=bs,
                statement_name_vi=bs_name,
                section="tai_san_ngan_han",
                aliases=[
                    "tiền và tương đương tiền", "tiền và các khoản tương đương tiền",
                    "tiền mặt và tương đương tiền",
                    "cash and cash equivalents",
                ],

            )
        )
        self._register(
            MetricDefinition(
                canonical_name="ĐẦU TƯ TÀI CHÍNH NGẮN HẠN",
                code="120",
                statement_type=bs,
                statement_name_vi=bs_name,
                section="tai_san_ngan_han",
                aliases=[
                    "đầu tư tài chính ngắn hạn", "chứng khoán kinh doanh", "đttc ngắn hạn",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="CÁC KHOẢN PHẢI THU NGẮN HẠN",
                code="130",
                statement_type=bs,
                statement_name_vi=bs_name,
                section="tai_san_ngan_han",
                aliases=[
                    "các khoản phải thu ngắn hạn", "phải thu ngắn hạn", "phải thu của khách hàng",
                    "phải thu khách hàng", "các khoản phải thu", "khoản phải thu ngắn hạn",
                    "accounts receivable",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="HÀNG TỒN KHO",
                code="140",
                statement_type=bs,
                statement_name_vi=bs_name,
                section="tai_san_ngan_han",
                aliases=[
                    "hàng tồn kho", "giá trị hàng tồn kho", "tồn kho", "dự phòng giảm giá hàng tồn kho",
                    "inventory",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="TÀI SẢN DÀI HẠN",
                code="200",
                statement_type=bs,
                statement_name_vi=bs_name,
                section="tai_san_dai_han",
                aliases=[
                    "tài sản dài hạn", "tổng tài sản dài hạn", "tsdh", "non-current assets",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="TÀI SẢN CỐ ĐỊNH",
                code="220",
                statement_type=bs,
                statement_name_vi=bs_name,
                section="tai_san_dai_han",
                aliases=[
                    "tài sản cố định", "tscđ", "tài sản cố định hữu hình", "tài sản cố định vô hình",
                    "fixed assets",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="TÀI SẢN THUẾ THU NHẬP HOÃN LẠI",
                code="262",
                statement_type=bs,
                statement_name_vi=bs_name,
                section="tai_san_dai_han",
                aliases=[
                    "tài sản thuế thu nhập hoãn lại",
                    "tài sản thuế hoãn lại",
                    "deferred tax assets",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="TỔNG NỢ PHẢI TRẢ",
                code="300",
                statement_type=bs,
                statement_name_vi=bs_name,
                section="no_phai_tra",
                aliases=[
                    "tổng nợ phải trả", "nợ phải trả", "tổng nợ", "các khoản nợ phải trả",
                    "các khoản nợ", "total liabilities", "liabilities",
                ],

            )
        )
        self._register(
            MetricDefinition(
                canonical_name="NỢ NGẮN HẠN",
                code="310",
                statement_type=bs,
                statement_name_vi=bs_name,
                section="no_phai_tra",
                aliases=[
                    "nợ ngắn hạn", "tổng nợ ngắn hạn", "các khoản nợ ngắn hạn", "nợ ngắn",
                    "current liabilities",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="PHẢI TRẢ NGƯỜI BÁN NGẮN HẠN",
                code="311",
                statement_type=bs,
                statement_name_vi=bs_name,
                section="no_phai_tra",
                aliases=[
                    "phải trả người bán ngắn hạn", "phải trả người bán", "phải trả nhà cung cấp",
                    "accounts payable",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="THUẾ VÀ CÁC KHOẢN PHẢI NỘP NHÀ NƯỚC",
                code="313",
                statement_type=bs,
                statement_name_vi=bs_name,
                section="no_phai_tra",
                aliases=[
                    "thuế và các khoản phải nộp nhà nước",
                    "thuế phải nộp nhà nước",
                    "taxes and other payables to the state",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="PHẢI TRẢ NGƯỜI LAO ĐỘNG",
                code="314",
                statement_type=bs,
                statement_name_vi=bs_name,
                section="no_phai_tra",
                aliases=[
                    "phải trả người lao động",
                    "khoản phải trả người lao động",
                    "payables to employees",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="VAY VÀ NỢ THUÊ TÀI CHÍNH NGẮN HẠN",
                code="320",
                statement_type=bs,
                statement_name_vi=bs_name,
                section="no_phai_tra",
                aliases=[
                    "vay và nợ thuê tài chính ngắn hạn", "vay ngắn hạn", "nợ vay ngắn hạn",
                    "vay nợ ngắn hạn", "short-term debt",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="NỢ DÀI HẠN",
                code="330",
                statement_type=bs,
                statement_name_vi=bs_name,
                section="no_phai_tra",
                aliases=[
                    "nợ dài hạn", "tổng nợ dài hạn", "các khoản nợ dài hạn", "nợ dài",
                    "long-term liabilities",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="VAY VÀ NỢ THUÊ TÀI CHÍNH DÀI HẠN",
                code="338",
                statement_type=bs,
                statement_name_vi=bs_name,
                section="no_phai_tra",
                aliases=[
                    "vay và nợ thuê tài chính dài hạn", "vay dài hạn", "nợ vay dài hạn",
                    "vay nợ dài hạn", "long-term debt",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="TỔNG VỐN CHỦ SỞ HỮU",
                code="400",
                statement_type=bs,
                statement_name_vi=bs_name,
                section="von_chu_so_huu",
                aliases=[
                    "tổng vốn chủ sở hữu", "vốn chủ sở hữu", "vcsh", "nguồn vốn chủ sở hữu",
                    "vốn chủ", "equity", "shareholders equity",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="VỐN ĐIỀU LỆ",
                code="411",
                statement_type=bs,
                statement_name_vi=bs_name,
                section="von_chu_so_huu",
                aliases=[
                    "vốn góp của chủ sở hữu", "vốn điều lệ", "vốn cổ phần", "vốn góp",
                    "charter capital", "paid-in capital",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="LỢI NHUẬN CHƯA PHÂN PHỐI",
                code="421",
                statement_type=bs,
                statement_name_vi=bs_name,
                section="von_chu_so_huu",
                aliases=[
                    "lợi nhuận sau thuế chưa phân phối", "lợi nhuận chưa phân phối",
                    "lãi chưa phân phối", "lncpp", "retained earnings",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="TIỀN GỬI CỦA KHÁCH HÀNG",
                code=None,
                statement_type=bs,
                statement_name_vi=bs_name,
                section="no_phai_tra",
                aliases=[
                    "tiền gửi của khách hàng", "tiền gửi khách hàng", "tiền gửi", "customer deposits",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="CHO VAY KHÁCH HÀNG",
                code=None,
                statement_type=bs,
                statement_name_vi=bs_name,
                section="tai_san",
                aliases=[
                    "cho vay khách hàng", "dư nợ cho vay", "dư nợ tín dụng", "cho vay",
                    "loans to customers",
                ],
            )
        )

        # 2. BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH
        is_type = StatementType.INCOME_STATEMENT
        is_name = STATEMENT_NAMES_VI[is_type]

        self._register(
            MetricDefinition(
                canonical_name="DOANH THU BÁN HÀNG VÀ CUNG CẤP DỊCH VỤ",
                code="01",
                statement_type=is_type,
                statement_name_vi=is_name,
                section="doanh_thu",
                aliases=[
                    "doanh thu bán hàng và cung cấp dịch vụ", "doanh thu bán hàng",
                    "tổng doanh thu", "doanh thu gộp", "gross revenue",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="CÁC KHOẢN GIẢM TRỪ DOANH THU",
                code="02",
                statement_type=is_type,
                statement_name_vi=is_name,
                section="doanh_thu",
                aliases=[
                    "các khoản giảm trừ doanh thu", "giảm trừ doanh thu",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="DOANH THU THUẦN",
                code="10",
                statement_type=is_type,
                statement_name_vi=is_name,
                section="doanh_thu",
                aliases=[
                    "doanh thu thuần về bán hàng và cung cấp dịch vụ",
                    "doanh thu thuần bán hàng và cung cấp dịch vụ",
                    "doanh thu thuần",
                    "doanh thu", "dtt", "doanh số", "net revenue", "revenue",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="GIÁ VỐN HÀNG BÁN",
                code="11",
                statement_type=is_type,
                statement_name_vi=is_name,
                section="chi_phi",
                aliases=[
                    "giá vốn hàng bán và dịch vụ cung cấp", "giá vốn hàng bán", "giá vốn",
                    "cogs", "cost of goods sold",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="LỢI NHUẬN GỘP",
                code="20",
                statement_type=is_type,
                statement_name_vi=is_name,
                section="loi_nhuan",
                aliases=[
                    "lợi nhuận gộp về bán hàng và cung cấp dịch vụ", "lợi nhuận gộp",
                    "lãi gộp", "lợi nhuận gộp về bán hàng", "gross profit",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="DOANH THU HOẠT ĐỘNG TÀI CHÍNH",
                code="21",
                statement_type=is_type,
                statement_name_vi=is_name,
                section="doanh_thu",
                aliases=[
                    "doanh thu hoạt động tài chính", "doanh thu tài chính", "lãi tiền gửi",
                    "financial income",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="CHI PHÍ TÀI CHÍNH",
                code="22",
                statement_type=is_type,
                statement_name_vi=is_name,
                section="chi_phi",
                aliases=[
                    "chi phí tài chính", "chi phí hđ tài chính", "financial expenses",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="CHI PHÍ LÃI VAY",
                code="23",
                statement_type=is_type,
                statement_name_vi=is_name,
                section="chi_phi",
                aliases=[
                    "chi phí lãi vay", "lãi vay", "chi phí lãi", "interest expense",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="CHI PHÍ BÁN HÀNG",
                code="25",
                statement_type=is_type,
                statement_name_vi=is_name,
                section="chi_phi",
                aliases=[
                    "chi phí bán hàng", "cpbh", "selling expenses",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="CHI PHÍ QUẢN LÝ DOANH NGHIỆP",
                code="26",
                statement_type=is_type,
                statement_name_vi=is_name,
                section="chi_phi",
                aliases=[
                    "chi phí quản lý doanh nghiệp", "chi phí quản lý", "cpqldn",
                    "general and administrative expenses", "g&a",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="LỢI NHUẬN THUẦN TỪ HOẠT ĐỘNG KINH DOANH",
                code="30",
                statement_type=is_type,
                statement_name_vi=is_name,
                section="loi_nhuan",
                aliases=[
                    "lợi nhuận thuần từ hoạt động kinh doanh", "lợi nhuận thuần",
                    "lợi nhuận hoạt động kinh doanh", "lnthđkd", "operating profit",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="TỔNG LỢI NHUẬN KẾ TOÁN TRƯỚC THUẾ",
                code="50",
                statement_type=is_type,
                statement_name_vi=is_name,
                section="loi_nhuan",
                aliases=[
                    "tổng lợi nhuận kế toán trước thuế", "lợi nhuận kế toán trước thuế",
                    "lợi nhuận trước thuế",
                    "lntt", "lãi trước thuế", "tổng lợi nhuận trước thuế", "profit before tax", "pbt",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="CHI PHÍ THUẾ TNDN",
                code="51",
                statement_type=is_type,
                statement_name_vi=is_name,
                section="chi_phi",
                aliases=[
                    "chi phí thuế thu nhập doanh nghiệp", "chi phí thuế tndn", "thuế tndn",
                    "thuế thu nhập doanh nghiệp", "chi phí thuế", "tax expense",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="CHI PHÍ THUẾ TNDN HIỆN HÀNH",
                code=None,
                statement_type=is_type,
                statement_name_vi=is_name,
                section="chi_phi_thue",
                aliases=[
                    "chi phí thuế thu nhập doanh nghiệp hiện hành",
                    "chi phí thuế tndn hiện hành",
                    "thuế thu nhập doanh nghiệp hiện hành",
                    "thuế tndn hiện hành",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="CHI PHÍ THUẾ TNDN HOÃN LẠI",
                code=None,
                statement_type=is_type,
                statement_name_vi=is_name,
                section="chi_phi_thue",
                aliases=[
                    "chi phí thuế thu nhập doanh nghiệp hoãn lại",
                    "chi phí thuế tndn hoãn lại",
                    "thuế thu nhập doanh nghiệp hoãn lại",
                    "thuế tndn hoãn lại",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="LỢI NHUẬN SAU THUẾ",
                code="60",
                statement_type=is_type,
                statement_name_vi=is_name,
                section="loi_nhuan",
                aliases=[
                    "lợi nhuận sau thuế thu nhập doanh nghiệp",
                    "lợi nhuận sau thuế tndn",
                    "lợi nhuận sau thuế",
                    "lnst", "lãi sau thuế", "lợi nhuận ròng", "lãi ròng", "lợi nhuận thuần sau thuế",
                    "net profit", "pat",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="LÃI CƠ BẢN TRÊN CỔ PHIẾU",
                code="70",
                statement_type=is_type,
                statement_name_vi=is_name,
                section="co_phieu",
                aliases=[
                    "lãi cơ bản trên cổ phiếu", "eps", "thu nhập trên mỗi cổ phiếu",
                    "thu nhập một cổ phiếu", "earnings per share",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="THU NHẬP LÃI THUẦN",
                code=None,
                statement_type=is_type,
                statement_name_vi=is_name,
                section="doanh_thu",
                aliases=[
                    "thu nhập lãi thuần", "lãi thuần", "thu nhập lãi", "net interest income", "nii",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="TỔNG THU NHẬP HOẠT ĐỘNG",
                code=None,
                statement_type=is_type,
                statement_name_vi=is_name,
                section="doanh_thu",
                aliases=[
                    "tổng thu nhập hoạt động", "thu nhập hoạt động", "toir", "operating income",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="THU NHẬP TỪ HOẠT ĐỘNG KHÁC",
                code=None,
                statement_type=is_type,
                statement_name_vi=is_name,
                section="doanh_thu_khac",
                aliases=[
                    "thu nhập từ hoạt động khác",
                    "thu nhập hoạt động khác",
                    "other operating income",
                ],
            )
        )

        # 3. BÁO CÁO LƯU CHUYỂN TIỀN TỆ
        cf = StatementType.CASH_FLOW
        cf_name = STATEMENT_NAMES_VI[cf]

        self._register(
            MetricDefinition(
                canonical_name="LƯU CHUYỂN TIỀN TỪ HOẠT ĐỘNG KINH DOANH",
                code="20",
                statement_type=cf,
                statement_name_vi=cf_name,
                section="dong_tien_kinh_doanh",
                aliases=[
                    "lưu chuyển tiền thuần từ hoạt động kinh doanh",
                    "lưu chuyển tiền thuần sử dụng từ hoạt động kinh doanh",
                    "tiền thuần từ hoạt động kinh doanh",
                    "lưu chuyển tiền từ hoạt động kinh doanh", "dòng tiền từ hoạt động kinh doanh",
                    "dòng tiền hoạt động kinh doanh", "dòng tiền thuần từ hoạt động kinh doanh",
                    "tiền từ hđkd", "dòng tiền kinh doanh", "lưu chuyển tiền từ hđkd", "cfo",
                    "cash flow from operations",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="LƯU CHUYỂN TIỀN TỪ HOẠT ĐỘNG ĐẦU TƯ",
                code="30",
                statement_type=cf,
                statement_name_vi=cf_name,
                section="dong_tien_dau_tu",
                aliases=[
                    "lưu chuyển tiền thuần từ hoạt động đầu tư",
                    "lưu chuyển tiền từ hoạt động đầu tư", "dòng tiền từ hoạt động đầu tư",
                    "dòng tiền hoạt động đầu tư", "dòng tiền đầu tư", "tiền từ hđđt",
                    "lưu chuyển tiền từ hđđt", "cfi",
                    "cash flow from investing",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="LƯU CHUYỂN TIỀN TỪ HOẠT ĐỘNG TÀI CHÍNH",
                code="40",
                statement_type=cf,
                statement_name_vi=cf_name,
                section="dong_tien_tai_chinh",
                aliases=[
                    "lưu chuyển tiền thuần từ hoạt động tài chính",
                    "lưu chuyển tiền từ hoạt động tài chính", "dòng tiền từ hoạt động tài chính",
                    "dòng tiền hoạt động tài chính", "dòng tiền tài chính", "tiền từ hđtc",
                    "lưu chuyển tiền từ hđtc", "cff",
                    "cash flow from financing",
                ],
            )
        )

        self._register(
            MetricDefinition(
                canonical_name="LƯU CHUYỂN TIỀN THUẦN TRONG KỲ",
                code="50",
                statement_type=cf,
                statement_name_vi=cf_name,
                section="dong_tien_tong",
                aliases=[
                    "lưu chuyển tiền thuần trong kỳ", "dòng tiền thuần", "biến động tiền thuần",
                    "tăng/giảm tiền thuần", "net cash flow",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="TIỀN VÀ TƯƠNG ĐƯƠNG TIỀN ĐẦU KỲ",
                code="60",
                statement_type=cf,
                statement_name_vi=cf_name,
                section="dong_tien_tong",
                aliases=[
                    "tiền và tương đương tiền đầu kỳ", "tiền và các khoản tương đương tiền đầu kỳ",
                    "tiền và các khoản tương đương tiền đầu năm", "tiền đầu năm", "tiền đầu kỳ",
                    "beginning cash",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="TIỀN VÀ TƯƠNG ĐƯƠNG TIỀN CUỐI KỲ",
                code="70",
                statement_type=cf,
                statement_name_vi=cf_name,
                section="dong_tien_tong",
                aliases=[
                    "tiền và tương đương tiền cuối kỳ", "tiền và các khoản tương đương tiền cuối kỳ",
                    "tiền và tương đương tiền cuối năm", "tiền và các khoản tương đương tiền cuối năm",
                    "tiền cuối năm", "tiền cuối kỳ", "ending cash",
                ],
            )
        )

        # 4. THUYẾT MINH BÁO CÁO TÀI CHÍNH
        notes = StatementType.NOTES
        notes_name = STATEMENT_NAMES_VI[notes]

        self._register(
            MetricDefinition(
                canonical_name="CHO VAY GIAO DỊCH KÝ QUỸ VÀ ỨNG TRƯỚC CHO KHÁCH HÀNG",
                code=None, statement_type=notes, statement_name_vi=notes_name,
                section="cho_vay_khach_hang",
                aliases=["cho vay giao dịch ký quỹ và ứng trước cho khách hàng"],
            )
        )

        self._register(
            MetricDefinition(
                canonical_name="THUYẾT MINH HÀNG TỒN KHO",
                code=None,
                statement_type=notes,
                statement_name_vi=notes_name,
                section="thuyet_minh",
                aliases=[
                    "thuyết minh hàng tồn kho", "tm hàng tồn kho", "chi tiết hàng tồn kho",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="THUYẾT MINH PHẢI THU KHÁCH HÀNG",
                code=None,
                statement_type=notes,
                statement_name_vi=notes_name,
                section="thuyet_minh",
                aliases=[
                    "thuyết minh phải thu", "tm phải thu", "chi tiết phải thu khách hàng",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="THUYẾT MINH NỢ PHẢI TRẢ",
                code=None,
                statement_type=notes,
                statement_name_vi=notes_name,
                section="thuyet_minh",
                aliases=[
                    "thuyết minh nợ phải trả", "tm nợ", "thuyết minh vay nợ",
                    "chi tiết vay và nợ thuê tài chính",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="THUYẾT MINH VỐN CHỦ SỞ HỮU",
                code=None,
                statement_type=notes,
                statement_name_vi=notes_name,
                section="thuyet_minh",
                aliases=[
                    "thuyết minh vốn chủ sở hữu", "tm vốn chủ", "biến động vốn chủ sở hữu",
                ],
            )
        )
        self._register(
            MetricDefinition(
                canonical_name="THUYẾT MINH DOANH THU",
                code=None,
                statement_type=notes,
                statement_name_vi=notes_name,
                section="thuyet_minh",
                aliases=[
                    "thuyết minh doanh thu", "tm doanh thu", "chi tiết doanh thu bán hàng",
                ],
            )
        )

    def _register(self, metric: MetricDefinition):
        if not metric.metric_id:
            metric.metric_id = self._build_metric_id(metric.canonical_name)
        if any(existing.metric_id == metric.metric_id for existing in self.metrics_list):
            raise ValueError(f"Duplicate financial metric id: {metric.metric_id}")
        if metric.canonical_name not in metric.aliases:
            metric.aliases.insert(0, metric.canonical_name)
        self.metrics_list.append(metric)
        for alias in metric.aliases:
            norm_alias = self.normalize_text(alias)
            self._alias_to_metric[norm_alias] = metric
        if metric.code:
            self._code_to_metric[(metric.statement_type, metric.code)] = metric


    @staticmethod
    def _build_metric_id(canonical_name: str) -> str:
        normalized = canonical_name.casefold().replace("đ", "d")
        ascii_name = unicodedata.normalize("NFKD", normalized).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"[^a-z0-9]+", "_", ascii_name).strip("_")

    @staticmethod
    def normalize_text(text: str) -> str:
        """Chuẩn hóa chuỗi văn bản: xóa số thứ tự đầu dòng, ký tự đặc biệt, lowercase."""
        t = text.strip().lower()
        # Loại bỏ tiền tố đánh số (ví dụ: "1. ", "A. ", "I. ", "10=01-02")
        t = re.sub(r"^\s*[a-z0-9ivx]+\s*[\.\:\-]\s*", "", t)
        t = re.sub(r"\([0-9\+\-\=\s\w]+\)$", "", t)
        t = re.sub(r"[^\w\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def lookup_metric(
        self,
        raw_text: str,
        code: Optional[str] = None,
        statement_hint: Optional[StatementType] = None,
    ) -> Optional[MetricDefinition]:
        """
        Tra cứu chỉ tiêu tài chính chuẩn hóa dựa trên chuỗi văn bản và/hoặc mã số.
        Ưu tiên nhãn mô tả đủ cụ thể trước mã biểu mẫu. Một số báo cáo ngân hàng
        dùng cùng mã dòng cho các khoản thuế hiện hành, hoãn lại và tổng hợp.
        """
        norm_text = self.normalize_text(raw_text)
        # 1. Tên chỉ tiêu chính xác
        exact_metric = self._alias_to_metric.get(norm_text) if norm_text else None
        if exact_metric is not None and (
            statement_hint is None
            or statement_hint == StatementType.GENERAL
            or exact_metric.statement_type == statement_hint
        ):
            return exact_metric

        # 2. Khớp substring dài nhất để giữ phân biệt tên chỉ tiêu con.
        matched_metric: Optional[MetricDefinition] = None
        longest_len = 0

        # Lọc danh sách ứng viên nếu có statement_hint
        candidates = (
            [m for m in self.metrics_list if m.statement_type == statement_hint]
            if statement_hint and statement_hint != StatementType.GENERAL
            else self.metrics_list
        )

        for m in candidates:
            for alias in m.aliases:
                norm_alias = self.normalize_text(alias)
                if len(norm_alias) >= 3 and (norm_alias in norm_text):
                    if len(norm_alias) > longest_len:
                        longest_len = len(norm_alias)
                        matched_metric = m

        if matched_metric is not None:
            return matched_metric

        # 3. Mã biểu mẫu chỉ là fallback khi OCR không còn nhãn mô tả.
        # Nhiều mẫu báo cáo, đặc biệt báo cáo ngân hàng và phần thuyết minh,
        # tái sử dụng cùng mã cho những dòng hoàn toàn khác nhau. Không dùng
        # mã để ghi đè một nhãn chữ chưa nhận diện được.
        has_descriptive_label = any(char.isalpha() for char in raw_text)
        if (
            not has_descriptive_label
            and code
            and statement_hint
            and (statement_hint, code.strip()) in self._code_to_metric
        ):
            return self._code_to_metric[(statement_hint, code.strip())]

        return None

    def detect_statement_from_query(self, query: str) -> Optional[StatementType]:
        """Xác định báo cáo tài chính từ câu hỏi người dùng dựa trên từ khóa phân hệ."""
        q_norm = query.lower()
        for stmt_type, keywords in STATEMENT_KEYWORD_ROUTING.items():
            if any(kw in q_norm for kw in keywords):
                return stmt_type
        return None

    def get_statement_name(self, statement_type: StatementType) -> str:
        return STATEMENT_NAMES_VI.get(statement_type, "Báo cáo tài chính")
