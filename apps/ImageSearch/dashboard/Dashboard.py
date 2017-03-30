import next.utils as utils
from next.apps.AppDashboard import AppDashboard
import pandas as pd
import numpy as np

class MyAppDashboard(AppDashboard):
    def __init__(self,db,ell):
        AppDashboard.__init__(self,db,ell)
        self.app_id = 'ImageSearch'

    def cumulative_reward_plot(self, app, butler):
        """
        Description: Returns multiline plot where there is a one-to-one mapping lines to
        algorithms and each line indicates the error on the validation set with respect to number of reported answers

        Expected input:
          None

        Expected output (in dict):
          (dict) MPLD3 plot dictionary
        """
        # get list of algorithms associated with project
        utils.debug_print('came into Dashboard')
        args = butler.experiment.get(key='args')
        num_algs = len(args['alg_list'])
        alg_labels = []
        for i in range(num_algs):
            alg_labels += [args['alg_list'][i]['alg_label']]

        # rewards = app.getModel(json.dumps({'exp_uid': app.exp_uid, 'args': {'alg_label': alg_label}}))

        # for algorithm in alg_labels:
        #     alg = utils.get_app_alg(self.app_id, algorithm)
        #     list_of_log_dict, didSucceed, message = butler.ell.get_logs_with_filter(app.app_id + ':ALG-EVALUATION',
        #                                                                             {'exp_uid': app.exp_uid,
        #                                                                              'alg_label': algorithm})
            # utils.debug_print('Trying to extract data for : ', algorithm)
            # list_of_log_dict, didSucceed, message = butler.ell.get_logs_with_filter(app.app_id, {'exp_uid': app.exp_uid,
            #                                                                           'alg_label': algorithm})
            # utils.debug_print('didSuceed, message', didSucceed, message)
            # utils.debug_print('app.expUID', app.exp_uid)
            # utils.debug_print('alg_label', algorithm)
            # utils.debug_print('list of log dict: ', list_of_log_dict)
            # data = alg.getModel(butler)
        plot_data = butler.dashboard.get(key='plot_data')
        # utils.debug_print('butler.algs.plot_data in Dashboard: ', plot_data)
        df = pd.DataFrame(plot_data)
        df.columns = [u'alg', u'arm_pulled', u'initial_arm', u'participant_uid', u'rewards', u'time']
        # utils.debug_print('df: ', df)
        # df = df.pivot_table(columns='initial arm', index='time', values='rewards', aggfunc=np.mean)
        # utils.debug_print('df: ', df)
        utils.debug_print('Came into Dashbord, trying to pring algs and init_arms')
        algs = list(df['alg'].unique())
        utils.debug_print('algs: ', algs)
        init_arms = df['initial_arm'].unique()
        utils.debug_print('init_arms: ', init_arms)
        import matplotlib.pyplot as plt
        import mpld3
        fig, ax = plt.subplots(nrows=1, ncols=len(init_arms), subplot_kw=dict(axisbg='#EEEEEE'))
        for i, init_arm in enumerate(init_arms):
            for alg in algs:
                print alg
                print init_arm
                result = df.query('alg == "{alg}" and initial_arm == {iarm}'.format(alg=alg, iarm=init_arm))[
                    ['time', 'rewards', 'participant_uid']].groupby('time').mean()
                rewards = np.array(result['rewards'])

                ax[i].plot(range(len(rewards)), np.cumsum(rewards), label='{alg}'.format(alg=alg))
                ax[i].set_xlabel('Time')
                ax[i].set_ylabel('Average cumulative rewards')
                # ax.set_xlim([x_min, x_max])
                # ax.set_ylim([y_min, y_max])
                # ax.grid(color='white', linestyle='solid')
                ax[i].set_title('Cumulative rewards for {sp}'.format(sp=init_arm), size=10)
                legend = ax[i].legend(loc=2, ncol=2, mode="expand")
                for label in legend.get_texts():
                    label.set_fontsize('xx-small')

        plot_dict = mpld3.fig_to_dict(fig)
        plt.close()
        return plot_dict
