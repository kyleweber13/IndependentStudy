import pingouin as pg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


class Data:

    def __init__(self, volume_file, agree_file):

        self.df_volume, self.df_agree = self.import_data(volume_file=volume_file, agree_file=agree_file)

    @staticmethod
    def import_data(volume_file, agree_file):

        df_volume = pd.read_excel(volume_file)
        df_agree = pd.read_excel(agree_file)

        return df_volume, df_agree

    def within_cutpoint_stats(self, show_plot=False):
        """Tests whether there are differences in measured activity volume between.
           Runs separate paired T-tests for differences between Powell (D vs. ND) and Esliger (R vs. L) cutpoint
           for each activity intensity.
        """

        n_subjs = len(set(self.df_volume["ID"]))
        t_crit = scipy.stats.t.ppf(.95, n_subjs - 1)
        ci_factor = t_crit / np.sqrt(n_subjs)

        # ----------------------------------------------- PAIRED T-TESTS ----------------------------------------------
        # Loops through and runs t-test for each intensity
        df_list = []

        for cutpoints in ["Powell", "Esliger"]:

            # Sets column names
            if cutpoints == "powell" or cutpoints == "Powell":
                nd = "Powell_ND"
                d = "Powell_D"
                epoch_len = 15
            if cutpoints == "esliger" or cutpoints == "Esliger":
                nd = "Esliger_L"
                d = "Esliger_R"
                epoch_len = 60

            for intensity_cat in ["Sedentary", "Light", "Moderate", "Vigorous"]:

                ttest = pg.ttest(x=self.df_volume.loc[(self.df_volume["EpochLen"] == epoch_len) &
                                                      (self.df_volume["Intensity"] == intensity_cat)][nd],
                                 y=self.df_volume.loc[(self.df_volume["EpochLen"] == epoch_len) &
                                                      (self.df_volume["Intensity"] == intensity_cat)][d],
                                 paired=True)

                ttest.insert(loc=0, column="Intensity", value=intensity_cat)
                ttest["Cutpoints"] = nd + "-" + d

                df_list.append(ttest)

            # df formatting. Sets intensity to index, drops BF10 column
            df_stats = pd.concat(df_list)
            df_stats = df_stats.set_index("Cutpoints", drop=True)
            df_stats = df_stats.drop("BF10", axis=1)

        # -------------------------------------------------- PLOTTING -------------------------------------------------
        if show_plot:

            powell_nd = self.df_volume.loc[self.df_volume["EpochLen"] == 15][["ID", "Intensity", "Powell_ND"]]
            powell_d = self.df_volume.loc[self.df_volume["EpochLen"] == 15][["ID", "Intensity", "Powell_D"]]

            esliger_l = self.df_volume.loc[self.df_volume["EpochLen"] == 60][["ID", "Intensity", "Esliger_L"]]
            esliger_r = self.df_volume.loc[self.df_volume["EpochLen"] == 60][["ID", "Intensity", "Esliger_R"]]

            fig = plt.subplots(2, 2, figsize=(10, 6))
            plt.subplots_adjust(hspace=.36)
            plt.suptitle("Within-Cutpoint Comparison (original epoch lengths)")

            plt.subplot(2, 2, 1)
            plt.title("Sedentary")

            plt.bar(x=["PowellND15", "PowellD15", "EsligerL60", "EsligerR60"],
                    height=[powell_nd.loc[powell_nd["Intensity"]=="Sedentary"].describe()["Powell_ND"]["mean"],
                            powell_d.loc[powell_d["Intensity"] == "Sedentary"].describe()["Powell_D"]["mean"],
                            esliger_l.loc[esliger_l["Intensity"] == "Sedentary"].describe()["Esliger_L"]["mean"],
                            esliger_r.loc[esliger_r["Intensity"] == "Sedentary"].describe()["Esliger_R"]["mean"]],

                    yerr=[powell_nd.loc[powell_nd["Intensity"] == "Sedentary"].describe()
                          ["Powell_ND"]["std"] * ci_factor,
                          powell_d.loc[powell_d["Intensity"] == "Sedentary"].describe()
                          ["Powell_D"]["std"] * ci_factor,
                          esliger_l.loc[esliger_l["Intensity"] == "Sedentary"].describe()
                          ["Esliger_L"]["std"] * ci_factor,
                          esliger_r.loc[esliger_r["Intensity"] == "Sedentary"].describe()
                          ["Esliger_R"]["std"] * ci_factor],
                    capsize=4, color=["red", "firebrick", "steelblue", "dodgerblue"], alpha=.7, edgecolor='black')
            plt.ylabel("% of data")

            plt.subplot(2, 2, 2)
            plt.title("Light")

            plt.bar(x=["PowellND15", "PowellD15", "EsligerL60", "EsligerR60"],
                    height=[powell_nd.loc[powell_nd["Intensity"] == "Light"].describe()["Powell_ND"]["mean"],
                            powell_d.loc[powell_d["Intensity"] == "Light"].describe()["Powell_D"]["mean"],
                            esliger_l.loc[esliger_l["Intensity"] == "Light"].describe()["Esliger_L"]["mean"],
                            esliger_r.loc[esliger_r["Intensity"] == "Light"].describe()["Esliger_R"]["mean"]],

                    yerr=[powell_nd.loc[powell_nd["Intensity"] == "Light"].describe()["Powell_ND"]["std"]*ci_factor,
                          powell_d.loc[powell_d["Intensity"] == "Light"].describe()["Powell_D"]["std"]*ci_factor,
                          esliger_l.loc[esliger_l["Intensity"] == "Light"].describe()["Esliger_L"]["std"]*ci_factor,
                          esliger_r.loc[esliger_r["Intensity"] == "Light"].describe()["Esliger_R"]["std"] * ci_factor],
                    capsize=4, color=["red", "firebrick", "steelblue", "dodgerblue"], alpha=.7, edgecolor='black')

            plt.subplot(2, 2, 3)
            plt.title("Moderate")

            plt.bar(x=["PowellND15", "PowellD15", "EsligerL60", "EsligerR60"],
                    height=[powell_nd.loc[powell_nd["Intensity"] == "Moderate"].describe()["Powell_ND"]["mean"],
                            powell_d.loc[powell_d["Intensity"] == "Moderate"].describe()["Powell_D"]["mean"],
                            esliger_l.loc[esliger_l["Intensity"] == "Moderate"].describe()["Esliger_L"]["mean"],
                            esliger_r.loc[esliger_r["Intensity"] == "Moderate"].describe()["Esliger_R"]["mean"]],

                    yerr=[powell_nd.loc[powell_nd["Intensity"] == "Moderate"].describe()["Powell_ND"]["std"]*ci_factor,
                          powell_d.loc[powell_d["Intensity"] == "Moderate"].describe()["Powell_D"]["std"]*ci_factor,
                          esliger_l.loc[esliger_l["Intensity"] == "Moderate"].describe()["Esliger_L"]["std"]*ci_factor,
                          esliger_r.loc[esliger_r["Intensity"] == "Moderate"].describe()["Esliger_R"]["std"]*ci_factor],
                    capsize=4, color=["red", "firebrick", "steelblue", "dodgerblue"], alpha=.7, edgecolor='black')
            plt.ylabel("% of data")

            plt.subplot(2, 2, 4)
            plt.title("Vigorous")

            plt.bar(x=["PowellND15", "PowellD15", "EsligerL60", "EsligerR60"],
                    height=[powell_nd.loc[powell_nd["Intensity"] == "Vigorous"].describe()["Powell_ND"]["mean"],
                            powell_d.loc[powell_d["Intensity"] == "Vigorous"].describe()["Powell_D"]["mean"],
                            esliger_l.loc[esliger_l["Intensity"] == "Vigorous"].describe()["Esliger_L"]["mean"],
                            esliger_r.loc[esliger_r["Intensity"] == "Vigorous"].describe()["Esliger_R"]["mean"]],

                    yerr=[powell_nd.loc[powell_nd["Intensity"] == "Vigorous"].describe()["Powell_ND"]["std"]*ci_factor,
                          powell_d.loc[powell_d["Intensity"] == "Vigorous"].describe()["Powell_D"]["std"]*ci_factor,
                          esliger_l.loc[esliger_l["Intensity"] == "Vigorous"].describe()["Esliger_L"]["std"]*ci_factor,
                          esliger_r.loc[esliger_r["Intensity"] == "Vigorous"].describe()["Esliger_R"]["std"]*ci_factor],
                    capsize=4, color=["red", "firebrick", "steelblue", "dodgerblue"], alpha=.7, edgecolor='black')

        return df_stats

    def across_epoch_stats(self, show_plot=False):
        """Tests whether scaling cutpoints to different epoch lengths affects measured activity volume.
           Runs separate paired T-tests on PowellND (15 vs. 60 seconds), PowellD (15 vs. 60 seconds),
           EsligerL (15 vs. 60 seconds), EsligerR (15 vs. 60 seconds) for each activity intensity
        """

        # ----------------------------------------------- PAIRED T-TESTS ----------------------------------------------
        # Loops through and runs t-test for each intensity
        df_list = []

        for cutpoints in ["Powell_ND", "Powell_D", "Esliger_L", "Esliger_R"]:
            for intensity_cat in ["Sedentary", "Light", "Moderate", "Vigorous"]:
                ttest = pg.ttest(x=self.df_volume.loc[(self.df_volume["EpochLen"] == 15) &
                                                      (self.df_volume["Intensity"] == intensity_cat)][cutpoints],
                                 y=self.df_volume.loc[(self.df_volume["EpochLen"] == 60) &
                                                      (self.df_volume["Intensity"] == intensity_cat)][cutpoints],
                                 paired=True)

                ttest.insert(loc=0, column="Intensity", value=intensity_cat)
                ttest["Cutpoints"] = cutpoints + "15-60"

                df_list.append(ttest)

        # df formatting. Sets intensity to index, drops BF10 column
        df_stats = pd.concat(df_list)
        df_stats = df_stats.set_index("Cutpoints", drop=True)
        df_stats = df_stats.drop("BF10", axis=1)

        return df_stats

    def agreement_stats(self, data_type="Kappa", show_plot=False):

        n_subjs = len(set(self.df_agree["ID"]))
        t_crit = scipy.stats.t.ppf(.95, n_subjs - 1)
        ci_factor = t_crit / np.sqrt(n_subjs)

        # WITHIN-AUTHOR -----------------------------------------------------------------------------------------------
        df_powell15 = self.df_agree.loc[(self.df_agree["EpochLen"] == 15) & (self.df_agree["Comparison"] == "Powell")]
        df_esliger60 = self.df_agree.loc[(self.df_agree["EpochLen"] == 60) & (self.df_agree["Comparison"] == "Esliger")]

        within_author_t = pg.ttest(df_powell15["Kappa"], df_esliger60[data_type], paired=True)
        within_author_t["Comparison"] = "Powell15(N-ND)-Esliger60(R-L)"
        within_author_t = within_author_t.set_index("Comparison", drop=True)

        if show_plot:

            fig = plt.subplots(1, 2, figsize=(10, 6))

            plt.subplot(1, 2, 1)

            plt.bar(x=["Powell N-ND", "Esliger R-L"],
                    height=[df_powell15.describe()["Kappa"]["mean"], df_esliger60.describe()["Kappa"]["mean"]],
                    yerr=[df_powell15.describe()["Kappa"]["std"]*ci_factor,
                          df_esliger60.describe()["Kappa"]["std"]*ci_factor],
                    capsize=4, color=['red', 'dodgerblue'], edgecolor='black', alpha=.7)

            plt.ylabel("Cohen's Kappa")
            plt.title("Cohen's Kappa")

            plt.subplot(1, 2, 2)

            plt.bar(x=["Powell N-ND", "Esliger R-L"],
                    height=[df_powell15.describe()["%Agree"]["mean"], df_esliger60.describe()["%Agree"]["mean"]],
                    yerr=[df_powell15.describe()["%Agree"]["std"] * ci_factor,
                          df_esliger60.describe()["%Agree"]["std"] * ci_factor],
                    capsize=4, color=['red', 'dodgerblue'], edgecolor='black', alpha=.7)

            plt.ylabel("% Agreement")
            plt.title("Percent Agreement")

        return within_author_t


# =====================================================================================================================
x = Data(agree_file="/Users/kyleweber/Desktop/OND05 Activity Data/All_Agreement.xlsx",
         volume_file="/Users/kyleweber/Desktop/OND05 Activity Data/All_ActivityVolume.xlsx")

# within_cutpoint_t = x.within_cutpoint_stats(True)

# within_cutpoint_agree = x.agreement_stats("Kappa", True)

# across_epochs_t = x.across_epoch_stats()
