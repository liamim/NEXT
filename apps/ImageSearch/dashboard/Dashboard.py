import json
from next.utils import utils
from next.apps.AppDashboard import AppDashboard
import next.apps.SimpleTargetManager
import pandas as pd
import numpy as np

class ImageSearchDashboard(AppDashboard):
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

        algs = list(df['alg'].unique())
        init_arms = df['initial_arm'].unique()
        import matplotlib.pyplot as plt
        import mpld3
        fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
        for init_arm in init_arms:
            for alg in algs:
                print alg
                print init_arm
                result = df.query('alg == "{alg}" and initial_arm == {iarm}'.format(alg=alg, iarm=init_arm))[
                    ['time', 'rewards', 'participant_uid']].groupby('time').mean()
                rewards = np.array(result['rewards'])

                ax.plot(range(len(rewards)), np.cumsum(rewards), label='Alg: {alg} and SP: {sp}'.format(alg=alg, sp=init_arm))

            # list_of_log_dict, didSucceed, message = butler.ell.get_logs_with_filter(app.app_id + ':ALG-EVALUATION',
            #                                                                         {'exp_uid': app.exp_uid,
            #                                                                          'alg_label': algorithm})
            # list_of_log_dict = sorted(list_of_log_dict, key=lambda item: utils.str2datetime(item['timestamp']))
            # utils.debug_print('list_of_log_dict: ', list_of_log_dict)
            # x = []
            # y = []
            # for item in list_of_log_dict:
            #     num_reported_answers = item['num_reported_answers']
            #     Xd = item['X']
            #     err = 0.5
            #     if len(test_S) > 0:
                    # compute error rate
                    # number_correct = 0.
                    # for query in test_S:
                    #     if 'q' in query:
                    #         i, j, k = query['q']
                    #         score = numpy.dot(Xd[j], Xd[j]) - 2 * numpy.dot(Xd[j], Xd[k]) + 2 * numpy.dot(Xd[i], Xd[
                    #             k]) - numpy.dot(Xd[i], Xd[i])
                    #         if score > 0:
                    #             number_correct += 1.0
                    #
                    # accuracy = number_correct / len(test_S)
                    # err = 1.0 - accuracy
                # x.append(num_reported_answers)
                # y.append(err)
            # alg_dict = {'legend_label': alg_label, 'x': x, 'y': y}
            # try:
            #     x_min = min(x_min, min(x))
            #     x_max = max(x_max, max(x))
            #     y_min = min(y_min, min(y))
            #     y_max = max(y_max, max(y))
            # except:
            #     pass
            # list_of_alg_dicts.append(alg_dict)
        #





        ax.set_xlabel('Time')
        ax.set_ylabel('Average cumulative rewards')
        # ax.set_xlim([x_min, x_max])
        # ax.set_ylim([y_min, y_max])
        # ax.grid(color='white', linestyle='solid')
        ax.set_title('Cumulative rewards', size=14)
        legend = ax.legend(loc=2, ncol=3, mode="expand")
        for label in legend.get_texts():
            label.set_fontsize('small')
        plot_dict = mpld3.fig_to_dict(fig)
        plt.close()
        return plot_dict
