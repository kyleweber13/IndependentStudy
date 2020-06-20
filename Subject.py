import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.metrics
import numpy as np
import os
from datetime import datetime
import matplotlib.dates as mdates
import warnings
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
warnings.filterwarnings("ignore")
os.chdir("/Users/kyleweber/Desktop/OND05 Activity Data/")

subjs = [10001, 10002, 10003, 10004, 10005, 10007, 10009, 10011, 10012, 20001, 20002, 20003, 20004, 20008]


class Subject:

    def __init__(self, data_folder=None, subject_id=None, nonwear_filepath=None, sleep_filepath=None, epoch_len=15):

        self.subjID = subject_id
        self.epoch_len = epoch_len
        self.data_folder = data_folder
        self.filename = ""
        self.nonwear_filepath = nonwear_filepath
        self.sleep_filepath = sleep_filepath

        self.df = None
        self.cutpoint_df = None
        self.intensity_df = None
        self.volume_df = None

        self.sleep_log = None
        self.nonwear_log = None

        self.df_agreement = None

    def create_filepathway(self):

        self.filename = self.data_folder + \
                        "{}-second epochs/{}_LWrist_{}.csv".format(self.epoch_len, self.subjID, self.epoch_len)

    def import_data(self):

        print("\nImporting epoched data ({})...".format(self.filename))

        self.df = pd.read_csv(filepath_or_buffer=self.filename, delimiter=",", skiprows=100, usecols=(0, 6, 7))
        self.df.columns = ["Timestamps", "Avg_Temp", "SVM"]

        # Converts from str to datetime
        self.df["Timestamps"] = pd.to_datetime(self.df["Timestamps"], format="%Y-%m-%d %H:%M:%S:%f")

        print("Complete.")

    def scale_cutpoints(self):

        powell_dom = [51 * 75 / 30, 68 * 75 / 30, 142 * 75 / 30]
        powell_nd = [47 * 75 / 30, 64 * 75 / 30, 157 * 75 / 30]

        esliger_l = [217 * 75/80, 644 * 75/80, 1810 * 75/80]
        esliger_r = [386 * 75/80, 439 * 75/80, 2098 * 75/80]

        if self.epoch_len == 15:
            esliger_l = [i/4 for i in esliger_l]
            esliger_r = [i/4 for i in esliger_r]

        if self.epoch_len == 60:
            powell_dom = [4*i for i in powell_dom]
            powell_nd = [4*i for i in powell_nd]

        self.cutpoint_df = pd.DataFrame(list(zip(powell_dom, powell_nd, esliger_l, esliger_r)),
                                        columns=["PowellDom", "PowellND", "EsligerL", "EsligerR"])

    def create_intensity_df(self):

        t0 = datetime.now()

        print("\nCalculating activity intensity...")

        epoch_intensities = []

        for svm, validity in zip([i for i in self.df["SVM"]], [i for i in self.df["Valid"]]):

            if validity == "Valid":

                if svm < self.cutpoint_df.iloc[0]["PowellND"]:
                    powell_nd_intensity = 0
                if self.cutpoint_df.iloc[0]["PowellND"] <= svm < self.cutpoint_df.iloc[1]["PowellND"]:
                    powell_nd_intensity = 1
                if self.cutpoint_df.iloc[1]["PowellND"] <= svm < self.cutpoint_df.iloc[2]["PowellND"]:
                    powell_nd_intensity = 2
                if self.cutpoint_df.iloc[2]["PowellND"] <= svm:
                    powell_nd_intensity = 3

                if svm < self.cutpoint_df.iloc[0]["PowellDom"]:
                    powell_d_intensity = 0
                if self.cutpoint_df.iloc[0]["PowellDom"] <= svm < self.cutpoint_df.iloc[1]["PowellDom"]:
                    powell_d_intensity = 1
                if self.cutpoint_df.iloc[1]["PowellDom"] <= svm < self.cutpoint_df.iloc[2]["PowellDom"]:
                    powell_d_intensity = 2
                if self.cutpoint_df.iloc[2]["PowellDom"] <= svm:
                    powell_d_intensity = 3

                if svm < self.cutpoint_df.iloc[0]["EsligerL"]:
                    esliger_l_intensity = 0
                if self.cutpoint_df.iloc[0]["EsligerL"] <= svm < self.cutpoint_df.iloc[1]["EsligerL"]:
                    esliger_l_intensity = 1
                if self.cutpoint_df.iloc[1]["EsligerL"] <= svm < self.cutpoint_df.iloc[2]["EsligerL"]:
                    esliger_l_intensity = 2
                if self.cutpoint_df.iloc[2]["EsligerL"] <= svm:
                    esliger_l_intensity = 3

                if svm < self.cutpoint_df.iloc[0]["EsligerR"]:
                    esliger_r_intensity = 0
                if self.cutpoint_df.iloc[0]["EsligerR"] <= svm < self.cutpoint_df.iloc[1]["EsligerR"]:
                    esliger_r_intensity = 1
                if self.cutpoint_df.iloc[1]["EsligerR"] <= svm < self.cutpoint_df.iloc[2]["EsligerR"]:
                    esliger_r_intensity = 2
                if self.cutpoint_df.iloc[2]["EsligerR"] <= svm:
                    esliger_r_intensity = 3

                epoch_intensities.append([powell_nd_intensity, powell_d_intensity,
                                          esliger_l_intensity, esliger_r_intensity])
            if validity == "Invalid":
                epoch_intensities.append(["Invalid", "Invalid", "Invalid", "Invalid"])

        self.intensity_df = pd.DataFrame(list(epoch_intensities), columns=["Powell_ND", "Powell_D",
                                                                           "Esliger_L", "Esliger_R"])

        # CALCULATES ACTIVITY VOLUMES ---------------------------------------------------------------------------------
        n_valid_epochs = self.df["Valid"].loc[self.df["Valid"]=="Valid"].shape[0]

        # Calculates number of epochs in each intensity
        powell_nd_totals = [len(self.intensity_df.loc[self.intensity_df["Powell_ND"] == 0]),
                            len(self.intensity_df.loc[self.intensity_df["Powell_ND"] == 1]),
                            len(self.intensity_df.loc[self.intensity_df["Powell_ND"] == 2]),
                            len(self.intensity_df.loc[self.intensity_df["Powell_ND"] == 3])]

        powell_d_totals = [len(self.intensity_df.loc[self.intensity_df["Powell_D"] == 0]),
                           len(self.intensity_df.loc[self.intensity_df["Powell_D"] == 1]),
                           len(self.intensity_df.loc[self.intensity_df["Powell_D"] == 2]),
                           len(self.intensity_df.loc[self.intensity_df["Powell_D"] == 3])]

        esliger_l_totals = [len(self.intensity_df.loc[self.intensity_df["Esliger_L"] == 0]),
                            len(self.intensity_df.loc[self.intensity_df["Esliger_L"] == 1]),
                            len(self.intensity_df.loc[self.intensity_df["Esliger_L"] == 2]),
                            len(self.intensity_df.loc[self.intensity_df["Esliger_L"] == 3])]

        esliger_r_totals = [len(self.intensity_df.loc[self.intensity_df["Esliger_R"] == 0]),
                            len(self.intensity_df.loc[self.intensity_df["Esliger_R"] == 1]),
                            len(self.intensity_df.loc[self.intensity_df["Esliger_R"] == 2]),
                            len(self.intensity_df.loc[self.intensity_df["Esliger_R"] == 3])]

        # Creates df of number of epochs in each intensity
        self.volume_df = pd.DataFrame(list(zip(powell_nd_totals, powell_d_totals,
                                               esliger_l_totals, esliger_r_totals)),
                                      columns=["Powell_ND", "Powell_D", "Esliger_L", "Esliger_R"],
                                      index=["Sedentary", "Light", "Moderate", "Vigorous"])

        # Converts number of epochs to % of valid epochs
        self.volume_df = self.volume_df * 100 / n_valid_epochs

        t1 = datetime.now()
        t = (t1 - t0).seconds

        print("Complete ({} seconds).".format(t))

    def plot_epoched(self):

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(9, 6))
        plt.subplots_adjust(bottom=.19)

        ax1.plot(self.df["Timestamps"], self.df["SVM"], color='black', label="SVM")
        ax1.set_ylabel("Counts per {} seconds".format(self.epoch_len))

        ax2.plot(self.df["Timestamps"], self.df["Avg_Temp"], color='red', label="Temperature")
        ax2.set_ylabel("Temperature (ºC)")

        xfmt = mdates.DateFormatter("%Y%b%d %H:%M:%S")
        locator = mdates.HourLocator(byhour=[0, 12], interval=1)

        ax2.xaxis.set_major_formatter(xfmt)
        ax2.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=8)

    def run_statistics(self):

        # COHEN'S KAPPA VALUES ----------------------------------------------------------------------------------------

        df = self.intensity_df.loc[self.intensity_df["Powell_D"] != "Invalid"]

        powell_kappa = sklearn.metrics.cohen_kappa_score(y1=[i for i in df["Powell_D"]],
                                                         y2=[i for i in df["Powell_ND"]])

        esliger_kappa = sklearn.metrics.cohen_kappa_score(y1=[i for i in df["Esliger_L"]],
                                                          y2=[i for i in df["Esliger_R"]])

        nd_kappa = sklearn.metrics.cohen_kappa_score(y1=[i for i in df["Powell_ND"]],
                                                                    y2=[i for i in df["Esliger_L"]])

        d_kappa = sklearn.metrics.cohen_kappa_score(y1=[i for i in df["Powell_D"]],
                                                    y2=[i for i in df["Esliger_R"]])

        kappas = [powell_kappa, esliger_kappa, nd_kappa, d_kappa]

        print("\nCOHEN'S KAPPA VALUES")
        print("-Powell kappa = {}".format(round(powell_kappa, 3)))
        print("-Esliger kappa = {}".format(round(esliger_kappa, 3)))
        print("-Non-dominant kappa = {}".format(round(nd_kappa, 3)))
        print("-Dominant kappa = {}".format(round(d_kappa, 3)))

        # PERCENT AGREEMENT ------------------------------------------------------------------------------------------
        powell_agree = [d == nd for d, nd in zip(df["Powell_D"], df["Powell_ND"])].count(True) / \
                       df.shape[0] * 100

        esliger_agree = [l == r for r, l in zip(df["Esliger_L"], df["Esliger_R"])].count(True) / \
                        df.shape[0] * 100

        nd_agree = [p_nd == e_l for p_nd, e_l in zip(df["Powell_ND"], df["Esliger_L"])].count(True) / \
                       df.shape[0] * 100

        d_agree = [p_d == e_r for p_d, e_r in zip(df["Powell_D"], df["Esliger_R"])].count(True) / \
                       df.shape[0] * 100

        agrees = [powell_agree, esliger_agree, nd_agree, d_agree]

        print("\nPERCENT AGREEMENT:")
        print("-Powell: Dom to Non-Dom = {}%".format(round(powell_agree, 1)))
        print("-Esliger: Right to Left = {}%".format(round(esliger_agree, 1)))
        print("-Non-dominant: Powell Non-Dom to Esliger Left = {}%".format(round(nd_agree, 1)))
        print("-Dominant: Powell Dom to Esliger Right = {}%".format(round(d_agree, 1)))

        # DF
        self.df_agreement = pd.DataFrame(list(zip(kappas, agrees)), columns=["Kappa", "%Agree"],
                                         index=["Powell", "Esliger", "NonDom", "Dom"])

    def valid_data(self):

        t0 = datetime.now()
        print("\nImporting sleep and non-wear data logs...")

        # IMPORTS LOGS -----------------------------------------------------------------------------------------------
        nonwear_log = pd.read_excel(self.nonwear_filepath)
        self.nonwear_log = nonwear_log.loc[nonwear_log["ID"] == self.subjID]
        self.nonwear_log["Off"] = pd.to_datetime(self.nonwear_log["Off"], format="%Y%b%d %H:%M")
        self.nonwear_log["On"] = pd.to_datetime(self.nonwear_log["On"], format="%Y%b%d %H:%M")

        sleep_log = pd.read_excel(self.sleep_filepath)
        self.sleep_log = sleep_log.loc[sleep_log["ID"] == self.subjID]
        self.sleep_log["Asleep"] = pd.to_datetime(self.sleep_log["Asleep"], format="%Y%b%d %H:%M")
        self.sleep_log["Awake"] = pd.to_datetime(self.sleep_log["Awake"], format="%Y%b%d %H:%M")

        t1 = datetime.now()
        t = (t1 - t0).seconds

        print("Complete ({} seconds)".format(t))

        # MARKS EPOCHS ------------------------------------------------------------------------------------------------
        print("\nMarking epochs as wear/non-wear and awake/asleep...")

        # Creates list of 0s corresponding to each epoch
        nonwear_status = np.zeros(self.df.shape[0])
        sleep_status = np.zeros(self.df.shape[0])

        # Loops epoch timestamps
        for i, epoch_stamp in enumerate(x.df["Timestamps"]):

            # Non-wear data
            for off, on in zip(self.nonwear_log["Off"], self.nonwear_log["On"]):
                if off <= epoch_stamp <= on:
                    nonwear_status[i] = 1
                    break
                else:
                    nonwear_status[i] = 0

            # Sleep data
            for asleep, awake in zip(self.sleep_log["Asleep"], self.sleep_log["Awake"]):
                if asleep <= epoch_stamp <= awake:
                    sleep_status[i] = 1
                    break
                else:
                    sleep_status[i] = 0

        # Adds data to self.df
        self.df["WearStatus"] = ["Wear" if i == 0 else "Nonwear" for i in nonwear_status]
        self.df["SleepStatus"] = ["Awake" if i == 0 else "Asleep" for i in sleep_status]
        self.df["Valid"] = ["Valid" if wear == "Wear" and sleep == "Awake" else "Invalid"
                            for wear, sleep in zip(x.df["WearStatus"], x.df["SleepStatus"])]

        t2 = datetime.now()
        t = (t2 - t1).seconds

        print("Complete ({} seconds)".format(t))

    def plot_data_validity(self):

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(9, 6))
        plt.subplots_adjust(bottom=.19)
        plt.suptitle("Participant {}".format(self.subjID))

        # Epoched wrist data ------------------------------------------------------------------------------------------
        ax1.plot(self.df["Timestamps"], self.df["SVM"], color='black', label="SVM")
        ax1.set_ylabel("Counts per {} seconds".format(self.epoch_len))

        # Epoched wear status -----------------------------------------------------------------------------------------
        ax2.plot(self.df["Timestamps"], self.df["WearStatus"], color='black')

        # Fills in non-wear periods
        for removal in range(self.nonwear_log.shape[0]):
            plt.fill_betweenx(x1=self.nonwear_log.iloc[removal]["Off"],
                              x2=self.nonwear_log.iloc[removal]["On"],
                              y=["Wear", "Nonwear"], color='grey', alpha=.5)

        # Sleep status  -----------------------------------------------------------------------------------------------
        ax2.plot(self.df["Timestamps"], self.df["SleepStatus"], color='black')

        # Fills in sleep periods
        for removal in range(self.sleep_log.shape[0]):
            plt.fill_betweenx(x1=self.sleep_log.iloc[removal]["Asleep"],
                              x2=self.sleep_log.iloc[removal]["Awake"],
                              y=["Asleep", "Awake"], color='blue', alpha=.5)

        # Validity ----------------------------------------------------------------------------------------------------
        ax2.plot(self.df["Timestamps"], self.df["Valid"], color='black')

        # Fills in invalid periods
        plt.fill_between(x=self.df["Timestamps"], y1="Invalid", y2="Valid", color='red', alpha=.5,
                         where=self.df["Valid"] == "Invalid")

        xfmt = mdates.DateFormatter("%Y%b%d %H:%M")
        locator = mdates.HourLocator(byhour=[0, 12], interval=1)

        ax2.xaxis.set_major_formatter(xfmt)
        ax2.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=8)

    def write_data(self):

        self.intensity_df.to_excel(self.filename.split("/")[-1].split(".")[0] + "_Intensity.xlsx")
        self.df_agreement.to_excel(self.filename.split("/")[-1].split(".")[0] + "_Agreement.xlsx")
        self.volume_df.to_excel(self.filename.split("/")[-1].split(".")[0] + "_ActivityVolume.xlsx")


x = Subject(subject_id=10001, data_folder="/Users/kyleweber/Desktop/Data/OND05/",
            epoch_len=15,
            sleep_filepath="/Users/kyleweber/Desktop/Data/OND05/OND05_SleepLog.xlsx",
            nonwear_filepath="/Users/kyleweber/Desktop/Data/OND05/OND05_NonwearLog.xlsx")

x.create_filepathway()
x.import_data()
x.scale_cutpoints()
x.valid_data()
# x.plot_epoched()
x.create_intensity_df()
x.run_statistics()
x.write_data()
