"""Build nested GCAM-China climate-ambition-risk-storage scenarios.

The ambition layer is upstream of the existing V3 risk/storage components. Each
scenario therefore carries one emissions pathway, one climate forcing case, one
extreme-risk state, and one storage pathway. The four retained combinations are
the minimum set needed to identify the risk penalty and the storage buffer.
"""

from __future__ import annotations

import csv
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
YEARS = tuple(range(2030, 2061, 5))
CLIMATES = ("ssp245", "ssp585")
PROVINCES = (
    "BJ", "TJ", "HE", "SX", "NM", "LN", "JL", "HL", "SH", "JS", "ZJ",
    "AH", "FJ", "JX", "SD", "HA", "HB", "HN", "GD", "GX", "HI", "CQ",
    "SC", "GZ", "YN", "XZ", "SN", "GS", "QH", "NX", "XJ",
)
CASES = (
    ("R0", "S1_ORDERLY", "optimized_r0_s1_orderly.xml"),
    ("R50", "S0_BASE", "optimized_r50_s0_base.xml"),
    ("R50", "S1_ORDERLY", "optimized_r50_s1_orderly.xml"),
    ("R50", "S2_ACCELERATED", "optimized_r50_s2_accelerated.xml"),
)


def read_constraints() -> dict[str, dict[int, float]]:
    path = ROOT / "data" / "v4" / "ambition_policy_constraints.csv"
    result: dict[str, dict[int, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result.setdefault(row["ambition_pathway"], {})[int(row["year"])] = float(
                row["co2_constraint_mtc"]
            )
    for pathway, values in result.items():
        missing = set(YEARS) - set(values)
        if missing:
            raise ValueError(f"{pathway} is missing constraint years: {sorted(missing)}")
    return result


def add_national_co2_market(tree: ET.ElementTree, constraints: dict[int, float]) -> None:
    world = tree.getroot().find("world")
    if world is None:
        raise ValueError("Scenario XML has no world node")

    provinces_seen = set()
    for region in world.findall("region"):
        name = region.get("name")
        if name not in PROVINCES:
            continue
        provinces_seen.add(name)
        for old_policy in list(region.findall("ghgpolicy")):
            if old_policy.get("name") == "CO2_CN":
                region.remove(old_policy)
        for old_link in list(region.findall("linked-ghg-policy")):
            if old_link.get("name") == "CO2" and old_link.findtext("linked-policy") == "CO2_CN":
                region.remove(old_link)
        provincial_policy = ET.SubElement(region, "ghgpolicy", {"name": "CO2_CN"})
        # GCAM-China removes the aggregate China region. A policy market hosted
        # in that deleted region fails during market ordering. The global market
        # container is therefore used only as a host; CO2_CN participation is
        # restricted to the 31 provinces, so the cap remains China-only.
        ET.SubElement(provincial_policy, "market").text = "global"
        if name == "BJ":
            for year in YEARS:
                if year == 2030:
                    # 2030 is the endogenous peak/calibration anchor, not an
                    # absolute cap. The official 15th FYP target is evaluated
                    # with intensity, non-fossil share, and peak diagnostics.
                    continue
                ET.SubElement(
                    provincial_policy, "constraint", {"year": str(year)}
                ).text = f"{constraints[year]:.10f}"
        linked = ET.SubElement(region, "linked-ghg-policy", {"name": "CO2"})
        ET.SubElement(linked, "price-adjust", {"fillout": "1", "year": "1975"}).text = "1"
        ET.SubElement(linked, "demand-adjust", {"fillout": "1", "year": "1975"}).text = "1"
        ET.SubElement(linked, "market").text = "global"
        ET.SubElement(linked, "linked-policy").text = "CO2_CN"
        ET.SubElement(linked, "price-unit").text = "1990$/tC"
        ET.SubElement(linked, "output-unit").text = "MtC"
    if provinces_seen != set(PROVINCES):
        raise ValueError(f"Province mismatch: {sorted(set(PROVINCES) - provinces_seen)}")


def set_config_value(root: ET.Element, name: str, value: str) -> None:
    matches = [node for node in root.iter("Value") if node.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one config Value named {name}, found {len(matches)}")
    matches[0].text = value


def write_tree(tree: ET.ElementTree, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="    ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def build_model_files(constraints: dict[str, dict[int, float]]) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    for climate in CLIMATES:
        source_dir = ROOT / "model_xml" / "v3" / climate
        out_dir = ROOT / "model_xml" / "v4_integrated" / climate
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        batch_root = ET.Element("BatchRunner")
        component_set = ET.SubElement(
            batch_root, "ComponentSet", {"name": "GCAM-China-RACA-integrated-climate-ambition"}
        )

        scenario_names = []
        for ambition, caps in constraints.items():
            for risk, storage, source_name in CASES:
                scenario = f"RACA_V4_{climate.upper()}_{ambition}_{risk}_{storage}"
                filename = f"{ambition.lower()}_{risk.lower()}_{storage.lower()}.xml"
                tree = ET.parse(source_dir / source_name)
                tree.getroot().set("name", scenario)
                add_national_co2_market(tree, caps)
                write_tree(tree, out_dir / filename)

                file_set = ET.SubElement(component_set, "FileSet", {"name": f"_{ambition}_{risk}_{storage}"})
                value = ET.SubElement(file_set, "Value", {"name": "raca_optimized_component"})
                value.text = f"../gcam_raca/model_xml/v4_integrated/{climate}/{filename}"
                scenario_names.append(scenario)
                manifest.append(
                    {
                        "scenario": scenario,
                        "climate_pathway": climate.upper(),
                        "ambition_pathway": ambition,
                        "risk_level": risk,
                        "storage_pathway": storage,
                        "component": str((out_dir / filename).relative_to(ROOT)),
                    }
                )

        batch_path = out_dir / f"batch_{climate}_12_integrated_scenarios.xml"
        write_tree(ET.ElementTree(batch_root), batch_path)

        config = ET.parse(source_dir / "configuration_optimized_12_scenarios_2060.xml")
        config_root = config.getroot()
        set_config_value(
            config_root,
            "BatchFileName",
            f"../gcam_raca/model_xml/v4_integrated/{climate}/{batch_path.name}",
        )
        set_config_value(
            config_root,
            "xmldb-location",
            f"../gcam_raca/outputs/v4/model_database/raca_v4_{climate}",
        )
        set_config_value(config_root, "scenarioName", f"RACA_V4_{climate.upper()}")
        write_tree(config, out_dir / "configuration_integrated_12_scenarios_2060.xml")

        query_root = ET.Element("ModelInterfaceBatch")
        clazz = ET.SubElement(query_root, "class", {"name": "ModelInterface.ModelGUI2.DbViewer"})
        command = ET.SubElement(clazz, "command", {"name": "XMLDB Batch File"})
        for scenario in scenario_names:
            ET.SubElement(command, "scenario", {"name": scenario})
        ET.SubElement(command, "queryFile").text = "gcam_raca/queries/raca_v4_integrated_queries.xml"
        ET.SubElement(command, "outFile").text = f"gcam_raca/outputs/v4/{climate}_query_results.csv"
        ET.SubElement(command, "xmldbLocation").text = f"gcam_raca/outputs/v4/model_database/raca_v4_{climate}"
        for tag, value in (
            ("batchQueryResultsInDifferentSheets", "false"),
            ("batchQueryIncludeCharts", "false"),
            ("batchQuerySplitRunsInDifferentSheets", "false"),
            ("batchQueryReplaceResults", "true"),
        ):
            ET.SubElement(command, tag).text = value
        write_tree(ET.ElementTree(query_root), ROOT / "queries" / f"query_v4_{climate}_batch.xml")

    return manifest


def write_launchers() -> None:
    for climate in CLIMATES:
        body = f'''@echo off
setlocal
set "JAVA_HOME=C:\\Program Files\\Java\\jre1.8.0_461"
set "PATH=%JAVA_HOME%\\bin;%JAVA_HOME%\\bin\\server;%PATH%"
cd /d "%~dp0..\\exe"
gcam.exe -C "..\\gcam_raca\\model_xml\\v4_integrated\\{climate}\\configuration_integrated_12_scenarios_2060.xml" > "..\\gcam_raca\\outputs\\v4\\{climate}_12_2060.log" 2>&1
if errorlevel 1 exit /b %errorlevel%
cd /d "%~dp0.."
"%JAVA_HOME%\\bin\\java.exe" -cp "libs\\jars\\*;output\\modelinterface\\ModelInterface.jar" ModelInterface.InterfaceMain -b "gcam_raca\\queries\\query_v4_{climate}_batch.xml" -l "gcam_raca\\outputs\\v4\\{climate}_query.log"
exit /b %errorlevel%
'''
        (ROOT / f"run_v4_{climate}.bat").write_text(body, encoding="ascii", newline="\r\n")


def build_smoke_test() -> None:
    climate = "ssp245"
    out_dir = ROOT / "model_xml" / "v4_integrated" / climate
    (ROOT / "outputs" / "v4" / "smoke").mkdir(parents=True, exist_ok=True)
    batch_root = ET.Element("BatchRunner")
    component_set = ET.SubElement(batch_root, "ComponentSet", {"name": "V4-integrated-smoke"})
    file_set = ET.SubElement(component_set, "FileSet", {"name": "_CN_NDC10_NZ2060_R0_S1_ORDERLY"})
    ET.SubElement(file_set, "Value", {"name": "raca_optimized_component"}).text = (
        "../gcam_raca/model_xml/v4_integrated/ssp245/cn_ndc10_nz2060_r0_s1_orderly.xml"
    )
    smoke_batch = out_dir / "batch_smoke_cn_ndc10_r0_s1.xml"
    write_tree(ET.ElementTree(batch_root), smoke_batch)

    config = ET.parse(out_dir / "configuration_integrated_12_scenarios_2060.xml")
    config_root = config.getroot()
    set_config_value(
        config_root,
        "BatchFileName",
        "../gcam_raca/model_xml/v4_integrated/ssp245/batch_smoke_cn_ndc10_r0_s1.xml",
    )
    set_config_value(
        config_root,
        "xmldb-location",
        "../gcam_raca/outputs/v4/smoke/model_database/raca_v4_smoke_ndc2035",
    )
    set_config_value(config_root, "scenarioName", "RACA_V4_SSP245_SMOKE")
    set_config_value(config_root, "stop-year", "2035")
    write_tree(config, out_dir / "configuration_smoke_cn_ndc10_r0_s1_2035.xml")

    body = '''@echo off
setlocal
set "JAVA_HOME=C:\\Program Files\\Java\\jre1.8.0_461"
set "PATH=%JAVA_HOME%\\bin;%JAVA_HOME%\\bin\\server;%PATH%"
cd /d "%~dp0..\\exe"
gcam.exe -C "..\\gcam_raca\\model_xml\\v4_integrated\\ssp245\\configuration_smoke_cn_ndc10_r0_s1_2035.xml" > "..\\gcam_raca\\outputs\\v4\\smoke\\gcam_smoke_ndc2035.log" 2>&1
exit /b %errorlevel%
'''
    (ROOT / "run_v4_smoke.bat").write_text(body, encoding="ascii", newline="\r\n")


def build_ipcc_resume() -> None:
    ambition = "IPCC15_ACCELERATED_2050"
    for climate in CLIMATES:
        out_dir = ROOT / "model_xml" / "v4_integrated" / climate
        batch_root = ET.Element("BatchRunner")
        component_set = ET.SubElement(batch_root, "ComponentSet", {"name": "V4-IPCC-resume"})
        for risk, storage, _ in CASES:
            filename = f"{ambition.lower()}_{risk.lower()}_{storage.lower()}.xml"
            file_set = ET.SubElement(
                component_set, "FileSet", {"name": f"_{ambition}_{risk}_{storage}"}
            )
            ET.SubElement(file_set, "Value", {"name": "raca_optimized_component"}).text = (
                f"../gcam_raca/model_xml/v4_integrated/{climate}/{filename}"
            )
        batch = out_dir / "batch_ipcc4_resume.xml"
        write_tree(ET.ElementTree(batch_root), batch)

        config = ET.parse(out_dir / "configuration_integrated_12_scenarios_2060.xml")
        config_root = config.getroot()
        set_config_value(
            config_root,
            "BatchFileName",
            f"../gcam_raca/model_xml/v4_integrated/{climate}/{batch.name}",
        )
        write_tree(config, out_dir / "configuration_ipcc4_resume_2060.xml")
        body = f'''@echo off
setlocal
set "JAVA_HOME=C:\\Program Files\\Java\\jre1.8.0_461"
set "PATH=%JAVA_HOME%\\bin;%JAVA_HOME%\\bin\\server;%PATH%"
cd /d "%~dp0..\\exe"
gcam.exe -C "..\\gcam_raca\\model_xml\\v4_integrated\\{climate}\\configuration_ipcc4_resume_2060.xml" > "..\\gcam_raca\\outputs\\v4\\{climate}_ipcc4_resume.log" 2>&1
if errorlevel 1 exit /b %errorlevel%
cd /d "%~dp0.."
"%JAVA_HOME%\\bin\\java.exe" -cp "libs\\jars\\*;output\\modelinterface\\ModelInterface.jar" ModelInterface.InterfaceMain -b "gcam_raca\\queries\\query_v4_{climate}_batch.xml" -l "gcam_raca\\outputs\\v4\\{climate}_query.log"
exit /b %errorlevel%
'''
        (ROOT / f"run_v4_{climate}_ipcc_resume.bat").write_text(
            body, encoding="ascii", newline="\r\n"
        )


def main() -> None:
    constraints = read_constraints()
    manifest = build_model_files(constraints)
    write_launchers()
    build_smoke_test()
    build_ipcc_resume()
    data_dir = ROOT / "data" / "v4"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "integrated_scenario_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (data_dir / "integrated_scenario_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Built {len(manifest)} nested scenarios across {len(CLIMATES)} climate pathways.")


if __name__ == "__main__":
    main()
