import torch


def drop(_X, _Y, _seq_len):
    if _X[-1].size(0) != _seq_len:
        _X, _Y = _X[:-1], _Y[:-1]
    return _X, _Y


def preprocess_data(_X, _Y, _X_test, _Y_test, _seq_len):
    # split into sequences
    _X = torch.split(_X, _seq_len, dim=0)
    _Y = torch.split(_Y, _seq_len, dim=0)

    _X_test = torch.split(_X_test, _seq_len, dim=0)
    _Y_test = torch.split(_Y_test, _seq_len, dim=0)

    _X, _Y = drop(_X, _Y, _seq_len)
    _X_test, _Y_test = drop(_X_test, _Y_test, _seq_len)

    return _X, _Y, _X_test, _Y_test



def preprocess_data_1(_X, _Y, _seq_len):
    # split into sequences
    _X = torch.split(_X, _seq_len, dim=0)
    _Y = torch.split(_Y, _seq_len, dim=0)


    _X, _Y = drop(_X, _Y, _seq_len)

    return _X, _Y



def plot_sound_events():
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    time_stamp_predict_file = "time_stamp_predict.txt"
    time_stamp_label_file = "time_stamp_label.txt"

    # Create a figure and a single subplot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors for different classes
    class_colors_dict = {
        'brakes squeaking': '#E41A1C',  # Red
        'car': '#377EB8',  # Blue
        'children': '#4DAF4A',  # Green
        'large vehicle': '#984EA3',  # Purple
        'people speaking': '#FF7F00',  # Orange
        'people walking': '#FFFF33'  # Yellow
    }

    column_names = ["start_time", "end_time", "class"]
    # Load the dataset
    data = pd.read_csv(time_stamp_predict_file, header=None, names=column_names)

    data_annotation = pd.read_csv(time_stamp_label_file, header=None, names=column_names)

    settings = {'annotated': {'y_value': 1, 'linestyle': '-', 'label': 'Annotated'},
                'model_output': {'y_value': 0.995, 'linestyle': '--', 'label': 'Model Output'}}
    # Closer y_value and different linestyle

    # Plot each class activity
    for _, row in data.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']
        audio_class = row['class']
        color = class_colors_dict[audio_class]

        ax.plot([start_time, end_time], [settings['annotated']['y_value'], settings['annotated']['y_value']],
                color=color,
                linewidth=10, linestyle=settings['annotated']['linestyle'])

    for _, row in data_annotation.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']
        audio_class = row['class']
        color = class_colors_dict[audio_class]

        ax.plot([start_time, end_time], [settings['model_output']['y_value'], settings['model_output']['y_value']],
                color=color,
                linewidth=10, linestyle=settings['model_output']['linestyle'])

        # Create legends
    class_legend = [mlines.Line2D([], [], color=color, label=audio_class) for audio_class, color in
                    class_colors_dict.items()]

    # data_legend = [mlines.Line2D([], [], color='black', linestyle=settings[data_type]['linestyle'],
    #                                 label=settings[data_type]['label']) for data_type in settings]

    # Add legends to the plot
    first_legend = ax.legend(handles=class_legend, title="Classes", loc='upper left')
    ax.add_artist(first_legend)
    # ax.legend(handles=data_legend, title="Data Type", loc='upper right')

    ax.set_yticks([])  # Hide Y axis
    ax.set_xlabel('Time (s)')
    ax.set_title('Audio Class Activity Over Time')
    ax.set_ylim(0.98, 1.02)

    plt.tight_layout()
    plt.savefig("ActivityBaselineTest.png")
    plt.show()
