import pingouin as pg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns


class Data:

    def __init__(self, volume_file, agree_file):

        self.df_volume, self.df_agree = self.import_data(volume_file=volume_file, agree_file=agree_file)

    @staticmethod
    def import_data(volume_file, agree_file):

        df_volume = pd.read_excel(volume_file)
        df_agree = pd.read_excel(agree_file)

        return df_volume, df_agree

    def valid_comparison(self, show_plot=True, test_type="parametric"):
        """Test of original parameters: PowellND15 vs. EsligerL60."""

        df_powell = self.df_volume.loc[self.df_volume["EpochLen"] == 15]
        df_powell = df_powell[["Intensity", "Powell_ND"]]
        df_esliger = self.df_volume.loc[self.df_volume["EpochLen"] == 60]
        df_esliger = df_esliger[["Intensity", "Esliger_L"]]

        df_list = []
        df_list_wtest = []
        df_list_shapiro = []

        for intensity in ["Sedentary", "Light", "Moderate", "Vigorous"]:

            ttest = pg.ttest(x=df_powell.loc[df_powell["Intensity"] == intensity]["Powell_ND"],
                             y=df_esliger.loc[df_esliger["Intensity"] == intensity]["Esliger_L"],
                             paired=True)

            wtest = scipy.stats.wilcoxon(x=df_powell.loc[df_powell["Intensity"] == intensity]["Powell_ND"],
                                         y=df_esliger.loc[df_esliger["Intensity"] == intensity]["Esliger_L"])

            shapiro = scipy.stats.shapiro(df_powell.loc[df_powell["Intensity"] == intensity]["Powell_ND"])
            shapiro2 = scipy.stats.shapiro(df_esliger.loc[df_esliger["Intensity"] == intensity]["Esliger_L"])

            df_list.append(ttest)
            df_list_wtest.append(wtest)
            df_list_shapiro.append(shapiro)
            df_list_shapiro.append(shapiro2)

        df_stats = pd.concat(df_list)
        df_stats["Intensity"] = ["Sedentary", "Light", "Moderate", "Vigorous"]
        df_stats = df_stats.set_index("Intensity", drop=True)
        df_stats["p < .05"] = ["*" if p < .05 else " " for p in df_stats["p-val"]]

        output = []
        for test in df_list_wtest:
            output.append([i for i in test])

        df_nonpara = pd.DataFrame(output, columns=["W", "p-val"], index=["Sedentary", "Light", "Moderate", "Vigorous"])
        df_nonpara["Intensity"] = ["Sedentary", "Light", "Moderate", "Vigorous"]
        df_nonpara = df_nonpara.set_index("Intensity", drop=True)
        df_nonpara["p < .05"] = ["*" if p < .05 else " " for p in df_nonpara["p-val"]]

        df_shapiro = pd.DataFrame(list(df_list_shapiro), columns=["W", "p-val"])
        df_shapiro["Intensity"] = ["Sedentary", "Sedentary", "Light", "Light",
                                   "Moderate", "Moderate", "Vigorous", "Vigorous"]
        df_shapiro["Data"] = ["Powell_ND", "Esliger_L", "Powell_ND", "Esliger_L",
                              "Powell_ND", "Esliger_L", "Powell_ND", "Esliger_L"]
        df_shapiro["p < .05"] = ["*" if p < .05 else " " for p in df_shapiro["p-val"]]
        df_shapiro = df_shapiro.set_index("Data", drop=True)

        # Plotting ----------------------------------------------------------------------------------------------------

        def plot_data(test_data="parametric"):

            if test_data == "parametric":
                sig_df = df_stats
            if test_data == "nonparametric" or test_type == "non-parametric":
                sig_df = df_nonpara

            n_subjs = len(set(self.df_volume["ID"]))
            t_crit = scipy.stats.t.ppf(.95, n_subjs - 1)
            ci_factor = t_crit / np.sqrt(n_subjs)

            plt.subplots(2, 2, figsize=(10, 7))
            plt.subplots_adjust(hspace=.28)
            plt.suptitle("Comparison of Original Cut-Point Parameters ({})".format(test_data))

            for i, intensity in enumerate(["Sedentary", "Light", "Moderate", "Vigorous"]):
                plt.subplot(2, 2, 1 + i)

                plt.bar(x=["Powell15_ND", "Esliger60_L"],
                        height=[df_powell.loc[df_powell["Intensity"] == intensity]["Powell_ND"].describe()["mean"],
                                df_esliger.loc[df_esliger["Intensity"] == intensity]["Esliger_L"].describe()['mean']],

                        yerr=[df_powell.loc[df_powell["Intensity"] == intensity]["Powell_ND"].describe()
                              ["std"] * ci_factor,
                              df_esliger.loc[df_esliger["Intensity"] == intensity]["Esliger_L"].describe()
                              ['std'] * ci_factor],
                        color=["red", "dodgerblue"], alpha=.7, edgecolor='black', capsize=4, width=.6)

                # Significance bars ------------------------------------------------------
                if sig_df.loc[intensity]["p-val"] < .05:
                    powell_h = df_powell.loc[df_powell["Intensity"] == intensity]["Powell_ND"].describe()["mean"] +\
                               df_powell.loc[df_powell["Intensity"] == intensity]["Powell_ND"].describe()['std'] \
                               * ci_factor
                    esliger_h = df_esliger.loc[df_esliger["Intensity"] == intensity]["Esliger_L"].describe()['mean'] +\
                                df_esliger.loc[df_esliger["Intensity"] == intensity]["Esliger_L"].describe()["std"] \
                                * ci_factor
                    powell_l = df_powell.loc[df_powell["Intensity"] == intensity]["Powell_ND"].describe()["mean"] -\
                               df_powell.loc[df_powell["Intensity"] == intensity]["Powell_ND"].describe()['std'] \
                               * ci_factor
                    esliger_l = df_esliger.loc[df_esliger["Intensity"] == intensity]["Esliger_L"].describe()['mean'] -\
                                df_esliger.loc[df_esliger["Intensity"] == intensity]["Esliger_L"].describe()["std"] \
                                * ci_factor

                    plt.plot(["Powell15_ND", "Powell15_ND", "Esliger60_L", "Esliger60_L"],
                             [max([powell_h, esliger_h])*1.1, max([powell_h, esliger_h])*1.15,
                              max([powell_h, esliger_h])*1.15, max([powell_h, esliger_h])*1.1], color='black')

                    # Label formatting
                    if round(sig_df.loc[intensity]["p-val"], 3) > .001:
                        plt.text(x=.5, y=max([powell_h, esliger_h]) * 1.2,
                                 s="p = {}".format(round(sig_df.loc[intensity]["p-val"], 3)),
                                 horizontalalignment="center")

                    if round(sig_df.loc[intensity]["p-val"], 3) <= .001:
                        plt.text(x=.5, y=max([powell_h, esliger_h]) * 1.2,
                                 s="p < .001",
                                 horizontalalignment="center")

                    # Y axis scale formatting
                    if min([powell_l, esliger_l]) > 0:
                        plt.ylim(0, max([powell_h, esliger_h])*1.3)

                    if min([powell_l, esliger_l]) <= 0:
                        plt.ylim(min([powell_l, esliger_l])*1.05, max([powell_h, esliger_h])*1.3)

                plt.title(intensity)

                if i == 0 or i == 2:
                    plt.ylabel("% of valid data")

        if show_plot:
            plot_data(test_data=test_type)

        return df_stats, df_nonpara, df_shapiro

    def epoch_scaling_comparison(self, show_plot=False, wear_side="nondom", test_type="parametric"):
        """Tests whether scaling cutpoints to different epoch lengths affects measured activity volume.
           Runs separate paired T-tests on PowellND (15 vs. 60 seconds), PowellD (15 vs. 60 seconds),
           EsligerL (15 vs. 60 seconds), EsligerR (15 vs. 60 seconds) for each activity intensity
        """

        if wear_side == "nondom" or wear_side == "NonDom" or wear_side.capitalize() == "Left":
            column_list = ["Powell_ND", "Esliger_L"]
            label_list = ["PowellND15", "PowellND60", "EsligerL15", "EsligerL60"]
            plot_title = "Non-dominant/Left"
        if wear_side == "mom" or wear_side == "Dom" or wear_side.capitalize() == "Right":
            column_list = ["Powell_D", "Esliger_R"]
            label_list = ["PowellD15", "PowellD60", "EsligerR15", "EsligerR60"]
            plot_title = "Dominant/Right"

        # ----------------------------------------------- PAIRED T-TESTS ----------------------------------------------
        # Loops through and runs t-test for each intensity
        df_list = []
        df_list_shapiro = []
        df_list_wtest = []

        for cutpoints in column_list:

            for intensity_cat in ["Sedentary", "Light", "Moderate", "Vigorous"]:
                ttest = pg.ttest(x=self.df_volume.loc[(self.df_volume["EpochLen"] == 15) &
                                                      (self.df_volume["Intensity"] == intensity_cat)][cutpoints],
                                 y=self.df_volume.loc[(self.df_volume["EpochLen"] == 60) &
                                                      (self.df_volume["Intensity"] == intensity_cat)][cutpoints],
                                 paired=True)

                wtest = scipy.stats.wilcoxon(x=self.df_volume.loc[(self.df_volume["EpochLen"] == 15) &
                                                                  (self.df_volume["Intensity"] == intensity_cat)][cutpoints],
                                             y=self.df_volume.loc[(self.df_volume["EpochLen"] == 60) &
                                                                  (self.df_volume["Intensity"] == intensity_cat)][cutpoints])

                ttest.insert(loc=0, column="Intensity", value=intensity_cat)
                ttest["Cutpoints"] = cutpoints + "15-60"

                df_list.append(ttest)

                shapiro = scipy.stats.shapiro(self.df_volume.loc[(self.df_volume["EpochLen"] == 15) &
                                                                 (self.df_volume["Intensity"] == intensity_cat)]
                                              [cutpoints])
                shapiro2 = scipy.stats.shapiro(self.df_volume.loc[(self.df_volume["EpochLen"] == 60) &
                                                                  (self.df_volume["Intensity"] == intensity_cat)]
                                               [cutpoints])

                df_list_shapiro.append(shapiro)
                df_list_shapiro.append(shapiro2)

                df_list_wtest.append(wtest)

        # df formatting. Sets intensity to index, drops BF10 column
        df_stats = pd.concat(df_list)
        df_stats = df_stats.set_index("Cutpoints", drop=True)
        df_stats = df_stats.drop("BF10", axis=1)
        df_stats["p < .05"] = ["*" if i < .05 else " " for i in df_stats["p-val"]]

        output = []
        for test in df_list_wtest:
            output.append([i for i in test])

        df_nonpara = pd.DataFrame(output, columns=["W", "p-val"])
        df_nonpara["Intensity"] = ["Sedentary", "Light", "Moderate", "Vigorous",
                                   "Sedentary", "Light", "Moderate", "Vigorous"]
        df_nonpara["Data"] = [column_list[0] + "15-60", column_list[0] + "15-60",
                              column_list[0] + "15-60", column_list[0] + "15-60",
                              column_list[1] + "15-60", column_list[1] + "15-60",
                              column_list[1] + "15-60", column_list[1] + "15-60"]
        df_nonpara = df_nonpara.set_index("Data", drop=True)
        df_nonpara["p < .05"] = ["*" if p < .05 else " " for p in df_nonpara["p-val"]]

        df_shapiro = pd.DataFrame(list(df_list_shapiro), columns=["W", "p-val"])
        df_shapiro["Intensity"] = ["Sedentary", "Sedentary", "Light", "Light",
                                   "Moderate", "Moderate", "Vigorous", "Vigorous",
                                   "Sedentary", "Sedentary", "Light", "Light",
                                   "Moderate", "Moderate", "Vigorous", "Vigorous"]
        df_shapiro["Data"] = [column_list[0] + "15-60", column_list[0] + "60",
                              column_list[0] + "15", column_list[0] + "60",
                              column_list[0] + "15", column_list[0] + "60",
                              column_list[0] + "15", column_list[0] + "60",
                              column_list[1] + "15", column_list[1] + "60",
                              column_list[1] + "15", column_list[1] + "60",
                              column_list[1] + "15", column_list[1] + "60",
                              column_list[1] + "15", column_list[1] + "60"]
        df_shapiro["p < .05"] = ["*" if p < .05 else " " for p in df_shapiro["p-val"]]
        df_shapiro = df_shapiro.set_index("Data", drop=True)

        # Plotting ----------------------------------------------------------------------------------------------------
        def plot_data(test_data):

            if test_data == "parametric":
                sig_df = df_stats
            if test_data == "nonparametric" or test_data == "non-parametric":
                sig_df = df_nonpara

            n_subjs = len(set(self.df_volume["ID"]))
            t_crit = scipy.stats.t.ppf(.95, n_subjs - 1)
            ci_factor = t_crit / np.sqrt(n_subjs)

            fig = plt.subplots(2, 2, figsize=(10, 7))
            plt.subplots_adjust(hspace=.33)

            plt.suptitle("Epoch Length Scaling within Author: {} data ({})".format(plot_title, test_data))

            for ind, intensity in enumerate(["Sedentary", "Light", "Moderate", "Vigorous"]):
                plt.subplot(2, 2, ind + 1)
                plt.title(intensity)

                plt.bar(x=label_list,
                        height=[self.df_volume.loc[(self.df_volume["Intensity"] == intensity) &
                                                   (self.df_volume["EpochLen"] == 15)].describe()
                                [column_list[0]]["mean"],
                                self.df_volume.loc[(self.df_volume["Intensity"] == intensity) &
                                                   (self.df_volume["EpochLen"] == 60)].describe()
                                [column_list[0]]["mean"],
                                self.df_volume.loc[(self.df_volume["Intensity"] == intensity) &
                                                   (self.df_volume["EpochLen"] == 15)].describe()
                                [column_list[1]]["mean"],
                                self.df_volume.loc[(self.df_volume["Intensity"] == intensity) &
                                                   (self.df_volume["EpochLen"] == 60)].describe()
                                [column_list[1]]["mean"]],
                        yerr=[self.df_volume.loc[(self.df_volume["Intensity"] == intensity) &
                                                 (self.df_volume["EpochLen"] == 15)].describe()[column_list[0]][
                                  "std"] * ci_factor,
                              self.df_volume.loc[(self.df_volume["Intensity"] == intensity) &
                                                 (self.df_volume["EpochLen"] == 60)].describe()[column_list[0]][
                                  "std"] * ci_factor,
                              self.df_volume.loc[(self.df_volume["Intensity"] == intensity) &
                                                 (self.df_volume["EpochLen"] == 15)].describe()[column_list[1]][
                                  "std"] * ci_factor,
                              self.df_volume.loc[(self.df_volume["Intensity"] == intensity) &
                                                 (self.df_volume["EpochLen"] == 60)].describe()[column_list[1]][
                                  "std"] * ci_factor],
                        color=["red", "red", "dodgerblue", "dodgerblue"],
                        edgecolor='black', alpha=.7, capsize=4)

                plt.xticks(fontsize=8)

                if ind == 0 or ind == 2:
                    plt.ylabel("% of data")

                # Significance bars ------------------------------
                stats_row = sig_df.loc[df_stats["Intensity"] == intensity]

                bottom, top = plt.ylim()

                # Used to set ylims
                sig_result = False

                # Powell data
                if stats_row.loc["Powell_ND15-60"]["p-val"] < .05:
                    sig_result = True
                    plt.plot(["PowellND15", "PowellND15", "PowellND60", "PowellND60"],
                             [top * 1.05, top * 1.1,
                              top * 1.1, top * 1.05], color='black')

                    if round(stats_row.loc["Powell_ND15-60"]["p-val"], 4) == 0:
                        plt.text(x=.5, y=top * 1.15, s="p < .001",
                                 horizontalalignment="center")

                    if round(stats_row.loc["Powell_ND15-60"]["p-val"], 4) >= 0:
                        plt.text(x=.5, y=top * 1.15, s="p = {}".format(round(stats_row.loc["Powell_ND15-60"]
                                                                            ["p-val"], 3)),
                                 horizontalalignment="center")

                # Esliger data
                if stats_row.loc["Esliger_L15-60"]["p-val"] < .05:
                    sig_result = True
                    plt.plot(["EsligerL15", "EsligerL15", "EsligerL60", "EsligerL60"],
                             [top * 1.05, top * 1.1,
                              top * 1.1, top * 1.05], color='black')

                    if round(stats_row.loc["Esliger_L15-60"]["p-val"], 4) == 0:
                        plt.text(x=2.5, y=top * 1.15, s="p < .001",
                                 horizontalalignment="center")

                    if round(stats_row.loc["Esliger_L15-60"]["p-val"], 4) >= 0:
                        plt.text(x=2.5, y=top * 1.15, s="p = {}".format(round(stats_row.loc["Esliger_L15-60"]
                                                                              ["p-val"], 3)),
                                 horizontalalignment="center")

                # Sets ylim if a result was significant
                if sig_result:
                    plt.ylim(bottom, top*1.3)

        if show_plot:
            plot_data(test_data=test_type)

        return df_stats, df_nonpara, df_shapiro

    def author_by_epoch(self, wear_side="nondom"):

        if wear_side == "nondom" or wear_side == "NonDom" or wear_side.capitalize() == "Left":
            column_list = ["Powell_ND", "Esliger_L"]
            plot_title = "Non-dominant/Left"
        if wear_side == "mom" or wear_side == "Dom" or wear_side.capitalize() == "Right":
            plot_title = "Dominant/Right"
            column_list = ["Powell_D", "Esliger_R"]

        df = self.df_volume[["ID", "EpochLen", "Intensity", column_list[0], column_list[1]]]

        anova_list = []

        plt.subplots(2, 2, figsize=(10, 7))
        plt.subplots_adjust(hspace=.28)
        plt.suptitle("Cutpoint x Epoch Interaction")

        colors = ["red", "#1E90FF"]
        sns.set_palette(sns.color_palette(colors))

        for i, intensity in enumerate(["Sedentary", "Light", "Moderate", "Vigorous"]):

            curr_df = df.loc[df["Intensity"] == intensity]

            curr_df = pd.melt(curr_df, id_vars=["ID", "EpochLen", "Intensity"],
                              value_vars=[column_list[0], column_list[1]],
                              var_name="Cutpoint", value_name="Percent")

            anova = pg.rm_anova(data=curr_df,
                                dv="Percent", within=["EpochLen", "Cutpoint"], subject="ID",
                                detailed=True, correction=True)

            anova["Intensity"] = [intensity, intensity, intensity]
            anova["p < .05"] = ["Significant" if i < .05 else "Not significant" for i in anova["p-unc"]]
            anova = anova.set_index("Intensity", drop=True)

            anova_list.append(anova)

            plt.subplot(2, 2, i + 1)

            bottom, top = plt.ylim()
            if intensity == "Sedentary":
                plt.ylim(0, 110)

            plt.title(intensity)
            sns.pointplot(x="EpochLen", y="Percent", hue="Cutpoint", data=curr_df, errwidth=1.5, capsize=.07)
            plt.legend()

            if i == 0 or i == 1:
                plt.xlabel("")
            if i == 1 or i == 3:
                plt.ylabel("")

        df_anova = pd.concat(anova_list)

        return df_anova

    def stop_here(self):
        pass

    def within_cutpoint_stats(self, show_plot=False):
        """Tests whether there are differences in measured activity volume.
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
                    height=[powell_nd.loc[powell_nd["Intensity"] == "Sedentary"].describe()["Powell_ND"]["mean"],
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

    def within_author_agreement_stats(self, data_type="Kappa", show_plot=False):
        """Tests whether the Powell15 ND-D agreement is different than the Esliger60 L-R agreement.
           Performs paired T-test.
        """

        n_subjs = len(set(self.df_agree["ID"]))
        t_crit = scipy.stats.t.ppf(.95, n_subjs - 1)
        ci_factor = t_crit / np.sqrt(n_subjs)

        # WITHIN-AUTHOR -----------------------------------------------------------------------------------------------
        df_powell15 = self.df_agree.loc[(self.df_agree["EpochLen"] == 15) & (self.df_agree["Comparison"] == "Powell")]
        df_esliger60 = self.df_agree.loc[(self.df_agree["EpochLen"] == 60) & (self.df_agree["Comparison"] == "Esliger")]

        within_author_t = pg.ttest(df_powell15[data_type], df_esliger60[data_type], paired=True)
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
            plt.ylim(0, 1)

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

    def between_cutpoint_stats(self, show_plot=True):

        powell = self.df_volume.loc[self.df_volume["EpochLen"] == 15][["ID", "Intensity", "Powell_ND"]]
        esliger = self.df_volume.loc[self.df_volume["EpochLen"] == 60][["ID", "Intensity", "Esliger_L"]]

        df_list = []

        for intensity in ["Sedentary", "Light", "Moderate", "Vigorous"]:

            ttest = pg.ttest(x=powell.loc[powell["Intensity"] == intensity]["Powell_ND"],
                             y=esliger.loc[esliger["Intensity"] == intensity]["Esliger_L"],
                             paired=True)

            ttest.insert(loc=0, column="Intensity", value=intensity)
            ttest["Cutpoints"] = "PowellND15-EsligerL60"

            df_list.append(ttest)

        df_stats = pd.concat(df_list)
        df_stats = df_stats.set_index("Cutpoints", drop=True)

        if show_plot:

            n_subjs = len(set(self.df_volume["ID"]))
            t_crit = scipy.stats.t.ppf(.95, n_subjs - 1)
            ci_factor = t_crit / np.sqrt(n_subjs)

            fig = plt.subplots(2, 2, figsize=(10, 7))

            for ind, intensity in enumerate(["Sedentary", "Light", "Moderate", "Vigorous"]):
                plt.subplot(2, 2, ind + 1)
                plt.title(intensity)

                plt.bar(x=["PowellND_15", "EsligerL_60"],
                        height=[self.df_volume.loc[(self.df_volume["EpochLen"] == 15) &
                                                   (self.df_volume["Intensity"] == intensity)].describe()
                                ["Powell_ND"]["mean"],
                                self.df_volume.loc[(self.df_volume["EpochLen"] == 60) &
                                                   (self.df_volume["Intensity"] == intensity)].describe()
                                ["Esliger_L"]["mean"]],
                        yerr=[self.df_volume.loc[(self.df_volume["EpochLen"] == 15) &
                                                   (self.df_volume["Intensity"] == intensity)].describe()
                                ["Powell_ND"]["std"] * ci_factor,
                                self.df_volume.loc[(self.df_volume["EpochLen"] == 60) &
                                                   (self.df_volume["Intensity"] == intensity)].describe()
                                ["Esliger_L"]["std"] * ci_factor],
                        color=["red", "steelblue"], edgecolor='black', alpha=.7, capsize=4)

                if ind == 0 or ind == 2:
                    plt.ylabel("% of data")

        return df_stats

    def between_author_agreement_stats(self, data_type="Kappa", show_plot=False):

        n_subjs = len(set(self.df_agree["ID"]))
        t_crit = scipy.stats.t.ppf(.95, n_subjs - 1)
        ci_factor = t_crit / np.sqrt(n_subjs)

        # WITHIN-AUTHOR -----------------------------------------------------------------------------------------------
        df = self.df_agree.loc[(self.df_agree["Comparison"] == "NonDom") | (self.df_agree["Comparison"] == "Dom")]

        anova = pg.rm_anova(data=df, dv=data_type, within=["EpochLen", "Comparison"], subject="ID", correction=True,
                            detailed=True)

        return anova


# =====================================================================================================================
x = Data(agree_file="/Users/kyleweber/Desktop/Data/OND05/OND05 Activity Data/All_Agreement.xlsx",
         volume_file="/Users/kyleweber/Desktop/Data/OND05/OND05 Activity Data/All_ActivityVolume.xlsx")

# Compares original parameters: Powell15ND vs. Esliger60L
original_para, original_nonpara, original_shapiro = x.valid_comparison(show_plot=True, test_type="nonparametric")

# Compares cut-points with an author scaled to other epoch length (activity volume)
epoch_para, epoch_nonpara, epoch_shapiro = x.epoch_scaling_comparison(wear_side="NonDom",
                                                                      show_plot=True, test_type="nonparametric")

# Compares cut-points from same author (activity volume)
# within_cutpoints_t = x.within_cutpoint_stats(show_plot=True)

# Compares [level of agreement within an author] between authors
# within_cutpoint_agree = x.within_author_agreement_stats("Kappa", True)

# Compares activity volume as measured by PowellND and EsligerL (correct cutpoints + epoch lengths)
# between_cutpoints_t = x.between_cutpoint_stats()

# anova = x.between_author_agreement_stats()
# interaction = x.author_by_epoch()
