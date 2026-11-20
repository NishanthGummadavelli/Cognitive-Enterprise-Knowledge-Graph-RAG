
"""Generate the initial eight-category enterprise classification dataset."""

import csv
from pathlib import Path
from src.classification.categories import CATEGORIES


# These are the eight finalized enterprise document categories.
TRAINING_EXAMPLES = {
    "HR": [
        "Employees are entitled to annual leave, sick leave and casual leave according to company policy.",
        "The recruitment process includes application screening, technical interviews and final HR approval.",
        "New employees must complete the onboarding process and submit their identification documents.",
        "The performance appraisal policy defines employee evaluation criteria and annual review procedures.",
        "Maternity leave applications must be submitted to the human resources department with supporting documents.",
        "The employee attendance policy explains working hours, late arrivals and leave regularization.",
        "Training and development programs are organized to improve employee skills and professional growth.",
        "The employee grievance procedure explains how workplace complaints are submitted and investigated.",
        "The compensation policy describes salary revisions, employee benefits and annual increments.",
        "Employees leaving the organization must complete the exit formalities and return company property.",
    ],
    "Finance": [
        "The quarterly financial statement summarizes revenue, operating expenses and net profit.",
        "All department purchase requests must be supported by an approved budget allocation.",
        "The accounts team reconciles bank statements with recorded financial transactions every month.",
        "Employee travel reimbursements require receipts and approval from the reporting manager.",
        "The annual budget outlines expected expenditure and revenue for each business department.",
        "Vendor invoices are verified against purchase orders before payment is processed.",
        "The cash flow report records incoming payments and outgoing operational expenses.",
        "Financial audits examine accounting records and verify compliance with internal controls.",
        "The procurement department must obtain finance approval before exceeding its spending limit.",
        "Tax records and supporting documents must be maintained for statutory financial reporting.",
    ],
    "IT": [
        "The deployment guide explains how to configure application servers and release software updates.",
        "Database administrators perform scheduled backups to protect enterprise information.",
        "Employees must use multi-factor authentication when accessing internal company applications.",
        "The incident response guide describes procedures for investigating server outages and security alerts.",
        "Network administrators monitor bandwidth usage and troubleshoot connectivity problems.",
        "The software maintenance document lists supported versions, patches and upgrade procedures.",
        "Developers must follow the version control workflow when submitting application code changes.",
        "The cloud infrastructure guide describes virtual machines, storage services and network configuration.",
        "Access permissions are reviewed regularly to protect confidential systems and data.",
        "The API integration manual describes authentication methods, request formats and response codes.",
    ],
    "Legal & Compliance": [
        "The service agreement defines the obligations of the provider and the customer.",
        "The confidentiality agreement restricts the disclosure of proprietary business information.",
        "Contract amendments must be reviewed and approved by the authorized legal representative.",
        "The compliance policy describes organizational responsibilities under applicable regulations.",
        "The intellectual property clause establishes ownership of software and original business materials.",
        "Disputes arising from the agreement will be handled according to the specified resolution procedure.",
        "The data protection agreement defines how personal information may be collected and processed.",
        "The vendor contract specifies termination conditions, liabilities and indemnification requirements.",
        "Employees must acknowledge the organization's code of conduct and applicable legal obligations.",
        "The regulatory filing document records the company's declarations to the relevant authorities.",
    ],
    "Operations": [
        "The standard operating procedure explains how warehouse inventory is received and inspected.",
        "Production supervisors must complete equipment safety checks before starting each shift.",
        "The delivery workflow defines order packing, dispatch scheduling and shipment tracking.",
        "The facility maintenance schedule assigns inspection tasks to the operations team.",
        "Quality control staff inspect finished products against established manufacturing standards.",
        "The customer service procedure describes how service requests are assigned and resolved.",
        "Inventory levels are reviewed weekly to prevent shortages of essential materials.",
        "The business continuity procedure defines operational steps during an unexpected disruption.",
        "The logistics team coordinates transportation schedules and distribution center activities.",
        "The equipment handling manual explains safe operating procedures for factory machinery.",
    ],
    "Sales & Marketing": [
        "The quarterly sales report summarizes product revenue, customer acquisitions and regional sales performance.",
        "The marketing strategy identifies target audiences, promotional channels and campaign objectives.",
        "The customer segmentation report groups buyers according to purchasing behavior and preferences.",
        "The advertising campaign plan defines promotional budgets, media schedules and expected reach.",
        "The sales proposal describes product offerings, pricing plans and commercial benefits for prospective clients.",
        "Market research examines competitor activities, customer demand and emerging industry trends.",
        "The lead management process explains how potential customers are identified and assigned to sales representatives.",
        "The brand guidelines define the approved use of company logos, messaging and promotional materials.",
        "The customer feedback report summarizes satisfaction ratings and suggestions for improving products.",
        "The sales forecast estimates future revenue based on current opportunities and historical performance.",
    ],
    "Projects & Management": [
        "The project charter defines the project objectives, scope, stakeholders and expected deliverables.",
        "The project progress report summarizes completed tasks, pending activities and current milestones.",
        "The meeting minutes record decisions, assigned responsibilities and follow-up actions.",
        "The project schedule identifies task dependencies, deadlines and resource allocation.",
        "The risk management plan identifies project risks and defines mitigation responsibilities.",
        "The management review evaluates team performance, project progress and delivery commitments.",
        "The resource allocation plan assigns employees, equipment and budgets to project activities.",
        "The project closure report summarizes completed deliverables, lessons learned and final approvals.",
        "The change request document describes proposed modifications to project scope and implementation timelines.",
        "The team coordination plan defines responsibilities, communication channels and reporting schedules.",
    ],
    "General & Administration": [
        "The organization was established to provide technology services to businesses across several industries.",
        "The company profile describes its history, mission statement and organizational background.",
        "The annual employee event celebrates team achievements and encourages workplace collaboration.",
        "The corporate newsletter shares general announcements and highlights recent organizational activities.",
        "The visitor information guide explains office locations, reception procedures and available facilities.",
        "The leadership message outlines the organization's long-term vision and general business values.",
        "The corporate overview introduces the company's major business units and international presence.",
        "The internal communication bulletin announces upcoming events and general workplace updates.",
        "The sustainability overview describes the organization's environmental goals and community initiatives.",
        "The organization chart presents the company's major divisions and reporting structure.",
    ],
}


def create_training_dataset(output_path: Path) -> None:
    """Write balanced synthetic examples and their source metadata to CSV.

    Args:
        output_path: Destination path for the generated CSV file.

    Raises:
        ValueError: If category examples are not balanced.
    """

    # Ensure the dataset uses exactly the official project categories.
    if set(TRAINING_EXAMPLES) != set(CATEGORIES):
    # Verify that all eight categories have equal example counts.
        raise ValueError("Training dataset categories do not match the official categories")
    
    example_counts = {
        category: len(examples)
        for category, examples in TRAINING_EXAMPLES.items()
    }

    if len(set(example_counts.values())) != 1:
        raise ValueError("Training examples must be balanced across categories")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Record provenance so synthetic and real examples can be distinguished.
    fieldnames = [
        "text",
        "label",
        "source_type",
        "source_name",
        "source_url",
        "license",
        "document_id",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for category, examples in TRAINING_EXAMPLES.items():
            for index, example in enumerate(examples, start=1):
                writer.writerow({
                    "text": example,
                    "label": category,
                    "source_type": "synthetic",
                    "source_name": "CodeCrafters synthetic dataset",
                    "source_url": "",
                    "license": "Project-created synthetic content",
                    "document_id": f"synthetic_{index:03d}_{category.lower().replace(' & ', '_').replace(' ', '_')}",
                })

    total_examples = sum(example_counts.values())

    print(f"Dataset created: {output_path}")
    print(f"Total examples: {total_examples}")
    print("Examples per category:")

    for category, count in example_counts.items():
        print(f"  {category}: {count}")


if __name__ == "__main__":
    dataset_path = Path("data/training/classification_dataset.csv")
    create_training_dataset(dataset_path)
