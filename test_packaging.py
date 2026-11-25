from importlib.resources import files
import bns_remnant_classifier


def test_package_has_version():
    assert bns_remnant_classifier.__version__ != "unknown"


def test_bundled_models_present():
    for name in ["classifierA_model.pkl", "classifierB_model.pkl",
                 "scaler_classifierA.pkl", "scaler_classifierB.pkl"]:
        assert files("bns_remnant_classifier").joinpath("models", name).is_file()


def test_bundled_default_eos_data_present():
    eos_prob = files("bns_remnant_classifier").joinpath(
        "eos_ns_properties_default", "eos_pos_setA.txt")
    mrl_dir = files("bns_remnant_classifier").joinpath(
        "eos_ns_properties_default", "MRL_default")
    assert eos_prob.is_file()
    assert mrl_dir.is_dir()
    # sanity: at least some .dat files actually shipped, not an empty dir
    assert any(p.name.endswith(".dat") for p in mrl_dir.iterdir())
