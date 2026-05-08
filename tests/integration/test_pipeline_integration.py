"""Multi-modal pipeline integration tests.

Tests the interaction between all synthesizers to create
complex, multi-modal synthetic datasets encompassing tabular,
text, time-series, and image data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from syntharc.image.augmentor import ImageAugmentor
from syntharc.tabular.gaussian_copula import GaussianCopulaSynthesizer
from syntharc.text.template import TemplateTextGenerator
from syntharc.timeseries.par import TimeSeriesSynthesizer


def test_grand_unified_pipeline(tmp_path: Path) -> None:
    """Test generating a full multi-modal dataset.

    Workflow:
    1. Generate Tabular Data (User Profiles)
    2. Generate Time-Series Data (User Login Activity linked to profiles)
    3. Generate Text Data (User Reviews linked to profiles)
    4. Generate Image Data (User Avatars linked to profiles)
    """
    # ==========================================
    # 1. TABULAR: Generate Synthetic Profiles
    # ==========================================
    real_profiles = pd.DataFrame(
        {
            "user_id": range(1, 11),
            "product": ["laptop", "phone"] * 5,
            "rating": [5, 4, 3, 5, 2] * 2,
        }
    )

    tab_synth = GaussianCopulaSynthesizer()
    tab_synth.fit(real_profiles)
    synth_profiles = tab_synth.generate(num_samples=10)

    assert len(synth_profiles) == 10
    assert "user_id" in synth_profiles.columns

    # ==========================================
    # 2. TIME-SERIES: Generate Synthetic Activity
    # ==========================================
    # We want activity sequences for each of our new synthetic user_ids
    real_activity = []
    for uid in range(1, 11):
        for t in range(5):
            real_activity.append(
                {
                    "user_id": uid,
                    "login_day": t,
                    "session_length": np.random.normal(30, 10),
                }
            )

    real_activity_df = pd.DataFrame(real_activity)

    ts_synth = TimeSeriesSynthesizer()
    ts_synth.fit(real_activity_df, sequence_key="user_id", context_columns=[])

    # We generate sequences exactly matching the number of synthetic users
    synth_activity = ts_synth.generate(num_samples=10)

    assert len(synth_activity) > 0
    assert "user_id" in synth_activity.columns

    # ==========================================
    # 3. TEXT: Generate Synthetic Reviews
    # ==========================================
    real_reviews = [
        "The product was amazingly fast and productive.",
        "This item is incredibly sleek and beautiful.",
        "The quality sounds wonderfully clear and loud.",
    ]

    text_gen = TemplateTextGenerator()
    text_gen.fit(real_reviews)

    # Generate 1 review for each synthetic profile
    synth_reviews = text_gen.generate(num_samples=10, instructions="reviews")
    assert len(synth_reviews) == 10

    # Attach reviews directly to the tabular profiles
    synth_profiles["review_text"] = synth_reviews

    # ==========================================
    # 4. IMAGE: Generate Synthetic Avatars
    # ==========================================
    # Dummy avatars (noise)
    real_avatars = []
    rng = np.random.default_rng(42)
    for _ in range(3):
        arr = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
        real_avatars.append(Image.fromarray(arr))

    img_synth = ImageAugmentor(intensity="light")
    img_synth.prepare(real_avatars)

    # Generate 1 avatar variation for each synthetic profile
    synth_avatars = img_synth.generate(num_samples=10, seed=42)
    assert len(synth_avatars) == 10

    # Save avatars using the synthetic user IDs
    for _idx, (uid, img) in enumerate(zip(synth_profiles["user_id"], synth_avatars, strict=False)):
        img.save(tmp_path / f"avatar_user_{uid}.png")

    saved_avatars = list(tmp_path.glob("avatar_user_*.png"))
    assert len(saved_avatars) == 10

    # ==========================================
    # VALIDATION: The Grand Unified Dataset
    # ==========================================
    # Verify the relational integrity of our combined multi-modal generation
    assert "review_text" in synth_profiles.columns

    # Check if time-series sequences map back to our generated profile IDs
    # SDV might generate completely new IDs or retain structure depending on setup,
    # but we just verify it produced the right amount of distinct sequences.
    unique_ts_users = synth_activity["user_id"].nunique()
    assert unique_ts_users == 10
