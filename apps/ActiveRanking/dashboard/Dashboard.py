import json
import next.utils as utils
from next.apps.AppDashboard import AppDashboard


class MyAppDashboard(AppDashboard):

    def __init__(self, db, ell):
        AppDashboard.__init__(self, db, ell)


    def current_queues(self, app, butler):
        alg_list = butler.experiment.get(key='args')['alg_list']        
        stats_data = {'headers': [{'label':'Alg Label', 'field':'alg_label'},
                                  {'label':'Available', 'field':'available'},
                                  {'label':'Query Queue Length', 'field':'queries'},
                                  {'label':'Queries Outstanding', 'field':'without_response'},
                                  {'label':'Queries Answered', 'field':'answered'}],
                      'plot_type': 'columnar_table',
                      'data':[]}

        plot_data = []
        for alg in alg_list:
            if alg['alg_id'].startswith('Quicksort'):
                logs, _, _ = butler.ell.get_logs_with_filter(butler.app_id+':ALG-EVALUATION',
                                                             {'exp_uid':butler.exp_uid,
                                                              'alg_label':alg['alg_label']})
                
                logs = sorted(logs, key=lambda item: utils.str2datetime(item['timestamp']) )
                plot_data.append({'legend_label':alg['alg_label'],
                                  'x':range(len(logs)), 'y':[x['queries'] for x in logs]})
                last = logs[-1]
                stats_data['data'].append({'alg_label':alg['alg_label'],
                                           'available':last['available'],
                                           'queries':last['queries'],
                                           'without_response':last['without_response'],
                                           'answered':last['answered']})
        import matplotlib.pyplot as plt
        import mpld3
        fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
        for alg in plot_data:
            ax.plot(alg['x'],alg['y'],label=alg['legend_label'])
        ax.set_xlabel('Number of answered queries')
        ax.set_ylabel('Size of Query Queue')
        #ax.set_xlim([0,])
        #ax.set_ylim([y_min,y_max])
        ax.grid(color='white', linestyle='solid')
        ax.set_title('Query Queue Sizes', size=14)
        legend = ax.legend(loc=2,ncol=3,mode="expand")
        for label in legend.get_texts():
          label.set_fontsize('small')
        plot = mpld3.fig_to_dict(fig)
        plt.close()
        return {'stats_data': stats_data, 'plot': plot}

    def most_current_ranking(self, app, butler, alg_label):
        """
        Description: Returns a ranking of items in the form of a list of dictionaries, 
        which is conveneint for downstream applications

        Expected input:
          (string) alg_label : must be a valid alg_label contained in alg_list list of dicts 

        The 'headers' contains a list of dictionaries corresponding to each column of the table with fields 'label' and 'field' 
        where 'label' is the label of the column to be put on top of the table, and 'field' is the name of the field in 'data' that the column correpsonds to 

        Expected output (in dict):
          plot_type : 'columnar_table'
          headers : [ {'label':'Rank','field':'rank'}, {'label':'Target','field':'index'} ]  
          (list of dicts with fields) data (each dict is a row, each field is the column for that row): 
            (int) index : index of target
            (int) ranking : rank (0 to number of targets - 1) representing belief of being best arm
        """
        item = app.getModel(json.dumps(
            {'exp_uid': app.exp_uid, 'args': {'alg_label': alg_label}}))

        targets = item['targets']
        #targets = sorted(targets, key=lambda x: x['rank'])
        return_dict = {}
        return_dict['headers'] = [{'label': 'Target', 'field': 'index'}, {
            'label': 'Rank', 'field': 'rank'}]
        return_dict['data'] = item['targets']
        return_dict['plot_type'] = 'columnar_table'
        return return_dict