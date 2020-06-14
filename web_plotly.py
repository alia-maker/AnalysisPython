from plotly import graph_objs as go
from plotly.offline import plot


def plotly_df(df, title = ''):
    data = []

    for column in df.columns:
        trace = go.Scatter(
            x=df.index,
            y=df[column],
            mode='lines',
            name=column
        )
        data.append(trace)

    layout = dict(title=title)
    fig = dict(data=data, layout=layout)
    plot(fig, show_link=False)

def plotly_df2(df, title = ''):
    # data = []
    # fig = go.Figure()
    #
    # fig.add_trace( go.Scatter(
    #     x=df.index,
    #     y=df["Real"],
    #     mode='lines',
    #     name="Actual",
    #     line=dict(color='blue', width=2.0)
    # ))
    # # data.append(trace)
    # fig.add_trace(go.Scatter(
    #     x=df.index,
    #     y=df["Model"],
    #     mode='lines',
    #     name="Model",
    #     line=dict(color='green', width=1.4)
    # ))
    #
    # # data.append(trace)
    #
    # fig.add_trace(go.Figure(
    #     x=df.index,
    #     y=df["UpperBond"],
    #     mode='lines',
    #     name="UpperBond",
    #     # line=dict(color='red', width=1),
    #     fill=None,
    # ))
    # # data.append(trace)
    #
    # fig.add_trace(go.Scatter(
    #     x=df.index,
    #     y=df["LowerBond"],
    #     mode='lines',
    #     name="LowerBond",
    #     # line=dict(color='red', width=1),
    #     fill='tonexty',
    # ))
    #
    # fig.update_layout(title_text=title)
    # # fig.show()
    # # data.append(trace)
    # # layout = dict(title=title)
    # # fig = dict(data=data, layout=layout)
    # plot(fig, show_link=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["UpperBond"],
                             fill=None,
                             mode='lines',
                             line_color='gray',
                             name="Upper bound"
                             ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["LowerBond"],
        fill='tonexty',  # fill area between trace0 and trace1
        name='Lower bound',
        mode='lines', line_color='gray'))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Real"],
        name="Actual",
        # fill='tonexty',  # fill area between trace0 and trace1
        mode='lines', line_color='blue'))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Model"],
        name="Model",
        # fill='tonexty',  # fill area between trace0 and trace1
        mode='lines', line_color='green'))

    fig.show()