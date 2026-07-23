"""The knowledge base's content: sector profiles, credit policy, and rendering.

One source of truth for the documents themselves, with no dependency on the
model or the vector store, so the same content serves three callers: the
chapter's seeding script, the tests, and the Lambda in aws/assistant-api, which
carries a copy of this file so it can seed itself on first call.

All content is synthetic. It reflects the shape of solar-sector corporate
lending without reproducing any lender's actual policy or any borrower's data.
"""

# -------------------------------------------------------------------------------
# Sector profiles
# -------------------------------------------------------------------------------
# Synthetic sector profiles for solar-sector corporate borrowers. Each becomes
# one knowledge-base document. The fields are what an underwriter weighs when
# there is no score to fall back on: how the business earns, what secures the
# loan, and where these companies tend to fail.
SECTORS = [
    {
        "slug": "residential-solar-installer",
        "sector": "Residential solar installer",
        "model": "Installs rooftop solar for homeowners, often with embedded financing at point of sale.",
        "cyclicality": "Sensitive to subsidy changes, interest rates, and electricity prices.",
        "margin": "Thin per install; volume and low customer-acquisition cost decide viability.",
        "leverage": "Working-capital heavy; funded debt to EBITDA under 3.0x preferred.",
        "dscr": "Floor 1.25x; test against a subsidy-cut scenario.",
        "collateral": "Receivables and inventory under a borrowing base; installed assets are the homeowner's.",
        "risks": "Subsidy and policy change, installer execution, warranty liability, thin balance sheet.",
        "profile": 7,
    },
    {
        "slug": "ci-solar-developer",
        "sector": "Commercial and industrial (C&I) solar developer",
        "model": "Develops and sometimes owns solar for businesses under long-term power purchase agreements.",
        "cyclicality": "Contracted cash flows dampen cycles; development stage is the risky part.",
        "margin": "Healthy once operating; development and interconnection costs are lumpy.",
        "leverage": "Project debt sized to contracted cash flow; corporate leverage kept modest.",
        "dscr": "Floor 1.30x on operating assets, tested against offtaker credit.",
        "collateral": "The solar asset, the PPA, and assignment of project accounts.",
        "risks": "Offtaker default, interconnection delay, construction overrun, curtailment.",
        "profile": 6,
    },
    {
        "slug": "battery-storage-integrator",
        "sector": "Battery storage integrator",
        "model": "Designs and installs battery systems, standalone or paired with solar.",
        "cyclicality": "Early market; revenue tied to subsidy and grid-service programs.",
        "margin": "Variable; hardware cost and program design drive returns.",
        "leverage": "Debt to EBITDA under 2.5x given revenue uncertainty.",
        "dscr": "Floor 1.35x; revenue-stacking assumptions discounted.",
        "collateral": "Equipment with uncertain resale; receivables where programs pay reliably.",
        "risks": "Technology and safety, program dependence, supplier concentration, degradation.",
        "profile": 8,
    },
    {
        "slug": "heat-pump-installer",
        "sector": "Heat pump installer",
        "model": "Installs residential and small commercial heat pumps, often subsidy-supported.",
        "cyclicality": "Seasonal and subsidy-driven; retrofit demand grows structurally.",
        "margin": "Thin per job; skilled-labor availability is the constraint.",
        "leverage": "Working-capital lines dominate; modest term debt.",
        "dscr": "Floor 1.25x.",
        "collateral": "Receivables and inventory; limited hard collateral.",
        "risks": "Subsidy timing, labor shortage, seasonality, workmanship claims.",
        "profile": 6,
    },
    {
        "slug": "solar-epc-contractor",
        "sector": "Solar EPC contractor",
        "model": "Engineering, procurement, and construction of solar projects for third-party owners.",
        "cyclicality": "Backlog and margins swing with the development pipeline.",
        "margin": "Contract-specific; overruns and retentions erase profit.",
        "leverage": "Modest funded debt; bonding capacity and off-balance items matter more.",
        "dscr": "Floor 1.35x; watch retention and unbilled work.",
        "collateral": "Receivables and equipment; contract retentions hard to realize.",
        "risks": "Contract disputes, cost overrun, working-capital swings, module supply.",
        "profile": 8,
    },
    {
        "slug": "solar-project-spv",
        "sector": "Solar project company (SPV)",
        "model": "A single-purpose company that owns an operating solar asset and its PPA.",
        "cyclicality": "Low once operating; cash flow is contracted for years.",
        "margin": "Predictable operating margin; sensitive to production and availability.",
        "leverage": "Project finance leverage, sized to the debt service coverage of the PPA.",
        "dscr": "Floor 1.30x on a P50 production case, 1.10x on a P90 case.",
        "collateral": "The asset, the PPA, and a full security package over project accounts.",
        "risks": "Production shortfall, offtaker default, O&M cost, refinancing at tail.",
        "profile": 5,
    },
    {
        "slug": "ev-charging-operator",
        "sector": "EV charging operator",
        "model": "Owns and operates public or fleet EV charging sites.",
        "cyclicality": "Utilization ramps slowly; early cash flows are thin.",
        "margin": "Negative to thin early; improves with utilization and pricing.",
        "leverage": "Underwritten on contracted or anchor-tenant revenue, not ramp assumptions.",
        "dscr": "Floor 1.40x on contracted revenue only.",
        "collateral": "Charging hardware with weak resale; site leases.",
        "risks": "Utilization risk, technology change, grid-connection cost, site concentration.",
        "profile": 8,
    },
    {
        "slug": "energy-efficiency-retrofit",
        "sector": "Energy efficiency retrofit firm",
        "model": "Delivers insulation, controls, and efficiency upgrades, often repaid from savings.",
        "cyclicality": "Counter-cyclical where energy prices are high; subsidy-linked.",
        "margin": "Modest; project sizing and measurement discipline matter.",
        "leverage": "Debt to EBITDA under 3.0x.",
        "dscr": "Floor 1.25x; savings-based repayment discounted.",
        "collateral": "Receivables; savings contracts are weak security.",
        "risks": "Measurement and verification, subsidy change, customer credit, execution.",
        "profile": 7,
    },
]

# -------------------------------------------------------------------------------
# Credit policy
# -------------------------------------------------------------------------------
# Synthetic credit policy. The first section is the one that changes for this
# book: with no external score, the risk profile is an underwriter's judgment,
# and the policy tells them how to form it and where it sends the decision.
POLICY = [
    {
        "id": "policy-risk-profile",
        "title": "Internal risk profile",
        "body": (
            "Solar-sector borrowers are young companies in a new market, so there"
            " is no external credit score. The underwriter assigns an internal"
            " risk profile from 1 to 10, where 10 is the most risky, using the"
            " five-factor rating scorecard that follows. The risk profile and the"
            " exposure together determine who may approve the deal."
        ),
    },
    {
        "id": "policy-rating-methodology",
        "title": "Rating methodology: the five-factor scorecard",
        "body": (
            "With no external score, the underwriter rates each deal on a"
            " qualitative scorecard adapted from the supervisory slotting approach"
            " for specialised lending. Five factors are each graded strong, good,"
            " satisfactory, or weak. One, financial strength: leverage, debt"
            " service coverage, liquidity, and the quality of projected cash flow."
            " Two, market and sector position: the sector profile, competitive"
            " position, and dependence on subsidies. Three, sponsor and execution:"
            " the track record of the installer or developer, management depth, and"
            " delivery history. Four, security package: collateral quality, the"
            " borrowing base, and for project companies the power purchase"
            " agreement and account security. Five, climate and transition:"
            " exposure to subsidy and policy change, physical and production risk"
            " on the assets, and the borrower's position in the energy transition."
            " The underwriter weights the factors, takes the weighted average, and"
            " maps it to a category. Strong maps to risk profile 1 to 3, good to 4"
            " to 5, satisfactory to 6 to 7, and weak to 8 to 10."
        ),
    },
    {
        "id": "policy-leverage",
        "title": "Leverage limits",
        "body": (
            "Funded debt to EBITDA is measured on a trailing twelve-month basis"
            " and, for development-stage borrowers, also against a conservative"
            " forward case. Standard maximum is 3.0x for operating companies."
            " Project companies are sized to debt service coverage, not EBITDA."
            " Exposures above the sector norm require approval one tier up."
        ),
    },
    {
        "id": "policy-dscr",
        "title": "Debt service coverage",
        "body": (
            "Debt service coverage ratio is cash flow available for debt service"
            " divided by scheduled principal and interest. The general floor is"
            " 1.25x. Project and PPA-backed exposures follow the sector floor in"
            " the relevant profile and are tested against a downside production"
            " and offtaker case. A ratio below the floor is a covenant breach."
        ),
    },
    {
        "id": "policy-collateral",
        "title": "Collateral coverage",
        "body": (
            "Secured facilities target collateral coverage of at least 1.5x for"
            " receivables and inventory under a borrowing base. Solar and battery"
            " equipment is discounted for weak resale value. For project"
            " companies, the security package is the asset, the power purchase"
            " agreement, and an assignment of project accounts."
        ),
    },
    {
        "id": "policy-concentration",
        "title": "Single-name concentration",
        "body": (
            "Total exposure to a single borrower group is capped at 10 percent of"
            " capital. Because the sector is young, exposure to a single subsidy"
            " program or a single offtaker is also monitored and reported."
        ),
    },
    {
        "id": "policy-authority",
        "title": "Delegated approval authority",
        "body": (
            "Every credit decision is taken at the lowest role whose exposure and"
            " risk-profile limits both cover the deal. A deal that exceeds the"
            " role on either dimension escalates to the next role. An underwriter"
            " decides exposures up to 2 million and risk profile up to 6. A"
            " managing director decides up to 10 million and profile up to 8. The"
            " chief executive decides any exposure and any profile."
        ),
    },
]

# -------------------------------------------------------------------------------
# Delegated authority tiers
# -------------------------------------------------------------------------------
# The delegated-authority matrix behind the routing tool. A startup escalates to
# people, not committees: underwriter, then managing director, then chief
# executive. Synthetic thresholds, in USD.
AUTHORITY_TIERS = [
    {
        "tier": 1,
        "name": "Underwriter",
        "max_exposure_usd": 2_000_000,
        "max_risk_profile": 6,
    },
    {
        "tier": 2,
        "name": "Managing director",
        "max_exposure_usd": 10_000_000,
        "max_risk_profile": 8,
    },
    {
        "tier": 3,
        "name": "Chief executive",
        "max_exposure_usd": None,
        "max_risk_profile": 10,
    },
]


# -------------------------------------------------------------------------------
# Document rendering and chunking
# -------------------------------------------------------------------------------
def sector_doc(item: dict) -> str:
    """Render one sector profile as a knowledge-base document."""
    return (
        f"# Sector profile: {item['sector']}\n\n"
        f"Business model: {item['model']}\n"
        f"Cyclicality: {item['cyclicality']}\n"
        f"Margin profile: {item['margin']}\n"
        f"Typical leverage: {item['leverage']}\n"
        f"DSCR expectation: {item['dscr']}\n"
        f"Collateral: {item['collateral']}\n"
        f"Key credit risks: {item['risks']}\n"
        f"Internal risk profile: {item['profile']} on a 1 to 10 scale, "
        f"where 10 is the most risky.\n"
    )


def policy_doc(item: dict) -> str:
    """Render one credit-policy section as a knowledge-base document."""
    return f"# Credit policy: {item['title']}\n\n{item['body']}\n"


def chunk_text(text: str, size: int = 600, overlap: int = 80) -> list[str]:
    """Split text into overlapping character windows."""
    step = size - overlap
    chunks = []
    for start in range(0, len(text), step):
        if piece := text[start : start + size].strip():
            chunks.append(piece)
        if start + size >= len(text):
            break
    return chunks


def documents() -> list[tuple[str, str]]:
    """Return (doc_id, text) for every knowledge-base document."""
    docs = [(f"kb/sector/{item['slug']}.md", sector_doc(item)) for item in SECTORS]
    docs += [(f"kb/policy/{item['id']}.md", policy_doc(item)) for item in POLICY]
    return docs
