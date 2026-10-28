"""
Dataset Generator for AI Sentiment Analyzer
===========================================
Generates a realistic, balanced, multi-domain sentiment dataset
exceeding 100 KB in size (2,200 rows, ~203 KB) with positive and negative reviews.
"""

import csv
import os
import random

POSITIVE_TEMPLATES = [
    "The {item} is absolutely {adj_pos} and {verb_pos} beyond expectations.",
    "I really love this {item}, the quality is {adj_pos} and it feels very {adj_pos2}.",
    "Exceptional {aspect} for this {item}, would definitely recommend to everyone.",
    "Very happy with the {item}, it arrived quickly and works {adv_pos}.",
    "Outstanding performance, this {item} is totally worth every penny.",
    "A truly {adj_pos} experience with the {item}, five stars all the way!",
    "The {aspect} of this {item} was top tier, thoroughly impressed with the craftsmanship.",
    "Superb customer service and the {item} delivered {adv_pos} results.",
    "I have been using this {item} for weeks and it remains completely {adj_pos}.",
    "Brilliant design, user-friendly interface, and {adj_pos} build quality.",
    "One of the best {item} purchases I have made this year, highly recommended.",
    "Delightful experience, the {item} is very {adj_pos} and extremely reliable.",
    "I was pleasantly surprised by how {adj_pos} the {item} turned out to be.",
    "Fabulous {aspect}, quick delivery, and the {item} looks gorgeous.",
    "The team was {adj_pos} and the {item} resolved all my issues seamlessly."
]

NEGATIVE_TEMPLATES = [
    "This {item} is completely {adj_neg} and stopped working after two days.",
    "Extremely disappointed with this {item}, absolute waste of money.",
    "Terrible quality, the {aspect} is {adj_neg} and cheaply constructed.",
    "Do not buy this {item}, customer service was rude and unresponsive.",
    "Horrible experience with the {item}, arrived broken and heavily damaged.",
    "The {item} failed to meet basic standards, highly {adj_neg} and annoying.",
    "Very frustrating experience, the {aspect} of this {item} is unacceptable.",
    "The {item} overheated constantly and performed {adv_neg} from day one.",
    "Defective parts, clumsy design, and totally {adj_neg} performance.",
    "I regret buying this {item}, it is practically useless and very flawed.",
    "Poor build quality, painful to use, and completely {adj_neg}.",
    "Dismal performance and worst {aspect} I have ever encountered.",
    "The {item} crashes frequently and customer support refused to refund.",
    "Awful experience from start to finish, the {item} is a disaster.",
    "Severely flawed {item}, sluggish interface, and cheap materials."
]

ITEMS = [
    "smartphone", "laptop", "wireless headphones", "smartwatch", "tablet",
    "bluetooth speaker", "gaming mouse", "mechanical keyboard", "coffee maker",
    "restaurant meal", "delivery service", "movie", "novel", "hotel stay",
    "airline flight", "fitness tracker", "running shoes", "winter jacket",
    "desk chair", "monitor display", "car rental", "software app", "camera lens"
]

ASPECTS = [
    "battery life", "audio clarity", "customer care", "shipping speed",
    "screen resolution", "comfort level", "taste and flavor", "durability",
    "build quality", "software stability", "response time", "ergonomics"
]

ADJ_POS = ["wonderful", "fantastic", "excellent", "stellar", "superb", "magnificent", "flawless", "impressive", "delightful", "outstanding"]
ADJ_POS2 = ["premium", "sturdy", "sleek", "reliable", "comfortable", "durable", "elegant", "solid"]
ADV_POS = ["flawlessly", "smoothly", "perfectly", "efficiently", "wonderfully", "admirably"]

ADJ_NEG = ["terrible", "horrible", "dreadful", "awful", "defective", "useless", "pathetic", "miserable", "faulty", "unreliable"]
ADV_NEG = ["terribly", "poorly", "horribly", "awfully", "dismally", "miserably"]


def generate_dataset(output_path="dataset.csv", target_rows=2200):
    rows = []
    random.seed(42)

    half = target_rows // 2
    for i in range(half):
        tmpl_p = random.choice(POSITIVE_TEMPLATES)
        txt_p = tmpl_p.format(
            item=random.choice(ITEMS),
            aspect=random.choice(ASPECTS),
            adj_pos=random.choice(ADJ_POS),
            adj_pos2=random.choice(ADJ_POS2),
            adv_pos=random.choice(ADV_POS),
            verb_pos="performed"
        )
        rows.append({"id": i * 2 + 1, "review": txt_p, "sentiment": "Positive", "label": 1})

        tmpl_n = random.choice(NEGATIVE_TEMPLATES)
        txt_n = tmpl_n.format(
            item=random.choice(ITEMS),
            aspect=random.choice(ASPECTS),
            adj_neg=random.choice(ADJ_NEG),
            adv_neg=random.choice(ADV_NEG)
        )
        rows.append({"id": i * 2 + 2, "review": txt_n, "sentiment": "Negative", "label": 0})

    random.shuffle(rows)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "review", "sentiment", "label"])
        writer.writeheader()
        writer.writerows(rows)

    size_bytes = os.path.getsize(output_path)
    size_kb = size_bytes / 1024.0
    print(f"Generated {output_path}: {len(rows)} rows, {size_bytes} bytes ({size_kb:.2f} KB)")
    return output_path, len(rows), size_kb


if __name__ == "__main__":
    generate_dataset()
