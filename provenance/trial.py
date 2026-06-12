import os

import numpy as np
import pandas as pd
import pooch

from relbench.base import Database, Dataset, Table
from relbench.utils import unzip_processor


class TrialDataset(Dataset):
    # 1 year gap
    val_timestamp = pd.Timestamp("2020-01-01")
    test_timestamp = pd.Timestamp("2021-01-01")

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""
        url = "https://relbench.stanford.edu/data/relbench-trial.zip"
        path = pooch.retrieve(
            url,
            known_hash="3f7376b7d901177157b3c5b048221884e936b45d05e809c7875403183ca9e13d",
            progressbar=True,
            processor=unzip_processor,
        )
        path = os.path.join(path, "relbench-trial-raw")
        studies = pd.read_csv(
            os.path.join(path, "studies.txt"), sep="|", low_memory=False
        )
        outcomes = pd.read_csv(os.path.join(path, "outcomes.txt"), sep="|")
        drop_withdrawals = pd.read_csv(
            os.path.join(path, "drop_withdrawals.txt"), sep="|"
        )
        designs = pd.read_csv(os.path.join(path, "designs.txt"), sep="|")
        eligibilities = pd.read_csv(os.path.join(path, "eligibilities.txt"), sep="|")
        interventions = pd.read_csv(
            os.path.join(path, "browse_interventions.txt"), sep="|"
        )
        interventions = interventions[
            interventions.mesh_type == "mesh-list"
        ]  # just looking at root identity
        conditions = pd.read_csv(os.path.join(path, "browse_conditions.txt"), sep="|")
        conditions = conditions[
            conditions.mesh_type == "mesh-list"
        ]  # just looking at root identity

        reported_event_totals = pd.read_csv(
            os.path.join(path, "reported_event_totals.txt"), sep="|"
        )
        sponsors = pd.read_csv(
            os.path.join(path, "sponsors.txt"), sep="|", low_memory=False
        )
        facilities = pd.read_csv(os.path.join(path, "facilities.txt"), sep="|")
        outcome_analyses = pd.read_csv(
            os.path.join(path, "outcome_analyses.txt"), sep="|", low_memory=False
        )
        detailed_descriptions = pd.read_csv(
            os.path.join(path, "detailed_descriptions.txt"), sep="|"
        )
        brief_summaries = pd.read_csv(
            os.path.join(path, "brief_summaries.txt"), sep="|"
        )

        ## just using trials with actual completion date
        # print('studies', len(studies))
        studies = studies[studies.completion_date_type == "Actual"]
        ## there are 27 trials before 1975
        studies = studies[studies.start_date >= "2000-01-01"]
        studies = studies[studies.nct_id.notnull()]
        # print('studies actual', len(studies))
        nct_id_use = studies.nct_id.values

        ## get trial start and end date for later infer
        studies["start_date"] = pd.to_datetime(studies["start_date"])
        studies["completion_date"] = pd.to_datetime(studies["completion_date"])
        nct2start_date = dict(studies[["nct_id", "start_date"]].values)
        nct2end_date = dict(studies[["nct_id", "completion_date"]].values)

        ## too many columns in studies, keeping few interesting columns and remove temporal leakage columns
        studies = studies[
            [
                "nct_id",
                "start_date",
                "target_duration",
                "study_type",
                "acronym",
                "baseline_population",
                "brief_title",
                "official_title",
                "phase",
                "enrollment",
                "enrollment_type",
                "source",
                "limitations_and_caveats",
                "number_of_arms",
                "number_of_groups",
                "has_dmc",
                "is_fda_regulated_drug",
                "is_fda_regulated_device",
                "is_unapproved_device",
                "is_ppsd",
                "is_us_export",
                "biospec_retention",
                "biospec_description",
                "source_class",
                "baseline_type_units_analyzed",
                "fdaaa801_violation",
                "plan_to_share_ipd",
            ]
        ]

        ## merge description/brief into main study table
        nct2descriptions = dict(detailed_descriptions[["nct_id", "description"]].values)
        nct2brief = dict(brief_summaries[["nct_id", "description"]].values)
        studies["detailed_descriptions"] = studies.nct_id.apply(
            lambda x: nct2descriptions[x] if x in nct2descriptions else np.nan
        )
        studies["brief_summaries"] = studies.nct_id.apply(
            lambda x: nct2brief[x] if x in nct2brief else np.nan
        )

        outcomes = outcomes[
            [
                "id",
                "nct_id",
                "outcome_type",
                "title",
                "description",
                "time_frame",
                "population",
                "units",
                "units_analyzed",
                "dispersion_type",
                "param_type",
            ]
        ]

        reported_event_totals = reported_event_totals[
            [
                "id",
                "nct_id",
                "event_type",
                "classification",
                "subjects_affected",
                "subjects_at_risk",
            ]
        ]

        drop_withdrawals.drop(
            columns=[
                "result_group_id",
                "ctgov_group_code",
                "drop_withdraw_comment",
                "reason_comment",
                "count_units",
            ],
            inplace=True,
        )
        conditions.drop(columns=["downcase_mesh_term", "mesh_type"], inplace=True)
        interventions.drop(columns=["downcase_mesh_term", "mesh_type"], inplace=True)
        ## filter to nct_id with actual completion date
        # print('outcomes before filter', len(outcomes))
        # for df in [outcomes, outcome_analyses, drop_withdrawals, reported_event_totals, designs, eligibilities, interventions, conditions, facilities, sponsors]:

        outcomes = outcomes[outcomes.nct_id.isin(nct_id_use)]
        outcome_analyses = outcome_analyses[outcome_analyses.nct_id.isin(nct_id_use)]
        drop_withdrawals = drop_withdrawals[drop_withdrawals.nct_id.isin(nct_id_use)]
        reported_event_totals = reported_event_totals[
            reported_event_totals.nct_id.isin(nct_id_use)
        ]
        designs = designs[designs.nct_id.isin(nct_id_use)]
        eligibilities = eligibilities[eligibilities.nct_id.isin(nct_id_use)]
        interventions = interventions[interventions.nct_id.isin(nct_id_use)]
        conditions = conditions[conditions.nct_id.isin(nct_id_use)]
        facilities = facilities[facilities.nct_id.isin(nct_id_use)]
        sponsors = sponsors[sponsors.nct_id.isin(nct_id_use)]

        # print('outcomes after filter', len(outcomes))
        ## infer time stamps
        ## tables that is available after trial ends
        for df in [outcomes, outcome_analyses, drop_withdrawals, reported_event_totals]:
            df["date"] = df.nct_id.apply(lambda x: nct2end_date[x])

        ## tables that is available as trial starts
        for df in [
            designs,
            eligibilities,
            interventions,
            conditions,
            facilities,
            sponsors,
        ]:
            df["date"] = df.nct_id.apply(lambda x: nct2start_date[x])

        ## create separate entity tables for sponsor/facility/condition/intervention since some tasks are asking them
        sponsor2id = dict(
            zip(sponsors.name.unique(), range(len(sponsors.name.unique())))
        )
        sponsors["sponsor_id"] = sponsors.name.apply(lambda x: sponsor2id[x])
        sponsor_trial = sponsors[
            ["id", "nct_id", "sponsor_id", "lead_or_collaborator", "date"]
        ]
        sponsors = (
            sponsors[["sponsor_id", "name", "agency_class"]]
            .drop_duplicates("sponsor_id")
            .reset_index(drop=True)
        )

        facility2id = dict(
            zip(facilities.name.unique(), range(len(facilities.name.unique())))
        )
        facilities["facility_id"] = facilities.name.apply(lambda x: facility2id[x])
        facility_trial = facilities[["id", "nct_id", "facility_id", "date"]]
        facilities = (
            facilities[["facility_id", "name", "city", "state", "zip", "country"]]
            .drop_duplicates("facility_id")
            .reset_index(drop=True)
        )

        condition2id = dict(
            zip(
                conditions.mesh_term.unique(), range(len(conditions.mesh_term.unique()))
            )
        )
        conditions["condition_id"] = conditions.mesh_term.apply(
            lambda x: condition2id[x]
        )
        condition_trial = conditions[["id", "nct_id", "condition_id", "date"]]
        conditions = (
            conditions[["condition_id", "mesh_term"]]
            .drop_duplicates("condition_id")
            .reset_index(drop=True)
        )

        intervention2id = dict(
            zip(
                interventions.mesh_term.unique(),
                range(len(interventions.mesh_term.unique())),
            )
        )
        interventions["intervention_id"] = interventions.mesh_term.apply(
            lambda x: intervention2id[x]
        )
        intervention_trial = interventions[["id", "nct_id", "intervention_id", "date"]]
        interventions = (
            interventions[["intervention_id", "mesh_term"]]
            .drop_duplicates("intervention_id")
            .reset_index(drop=True)
        )

        tables = {}

        tables["studies"] = Table(
            df=studies,
            fkey_col_to_pkey_table={},
            pkey_col="nct_id",
            time_col="start_date",
        )

        tables["outcomes"] = Table(
            df=outcomes,
            fkey_col_to_pkey_table={
                "nct_id": "studies",
            },
            pkey_col="id",
            time_col="date",
        )

        tables["outcome_analyses"] = Table(
            df=outcome_analyses,
            fkey_col_to_pkey_table={"nct_id": "studies", "outcome_id": "outcomes"},
            pkey_col="id",
            time_col="date",
        )

        tables["drop_withdrawals"] = Table(
            df=drop_withdrawals,
            fkey_col_to_pkey_table={
                "nct_id": "studies",
            },
            pkey_col="id",
            time_col="date",
        )

        tables["reported_event_totals"] = Table(
            df=reported_event_totals,
            fkey_col_to_pkey_table={
                "nct_id": "studies",
            },
            pkey_col="id",
            time_col="date",
        )

        tables["designs"] = Table(
            df=designs,
            fkey_col_to_pkey_table={
                "nct_id": "studies",
            },
            pkey_col="id",
            time_col="date",
        )

        tables["eligibilities"] = Table(
            df=eligibilities,
            fkey_col_to_pkey_table={
                "nct_id": "studies",
            },
            pkey_col="id",
            time_col="date",
        )

        tables["interventions"] = Table(
            df=interventions,
            fkey_col_to_pkey_table={},
            pkey_col="intervention_id",
            time_col=None,
        )

        tables["conditions"] = Table(
            df=conditions,
            fkey_col_to_pkey_table={},
            pkey_col="condition_id",
            time_col=None,
        )

        tables["facilities"] = Table(
            df=facilities,
            fkey_col_to_pkey_table={},
            pkey_col="facility_id",
            time_col=None,
        )

        tables["sponsors"] = Table(
            df=sponsors,
            fkey_col_to_pkey_table={},
            pkey_col="sponsor_id",
            time_col=None,
        )

        tables["interventions_studies"] = Table(
            df=intervention_trial,
            fkey_col_to_pkey_table={
                "nct_id": "studies",
                "intervention_id": "interventions",
            },
            pkey_col="id",
            time_col="date",
        )

        tables["conditions_studies"] = Table(
            df=condition_trial,
            fkey_col_to_pkey_table={"nct_id": "studies", "condition_id": "conditions"},
            pkey_col="id",
            time_col="date",
        )

        tables["facilities_studies"] = Table(
            df=facility_trial,
            fkey_col_to_pkey_table={"nct_id": "studies", "facility_id": "facilities"},
            pkey_col="id",
            time_col="date",
        )

        tables["sponsors_studies"] = Table(
            df=sponsor_trial,
            fkey_col_to_pkey_table={"nct_id": "studies", "sponsor_id": "sponsors"},
            pkey_col="id",
            time_col="date",
        )
        return Database(tables)
