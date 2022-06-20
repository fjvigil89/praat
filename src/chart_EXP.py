    
import plotly.graph_objects as go
import plotly.offline as pyo

if __name__ == '__main__':   
    columns= ['meanF0','stdevF0','localJitter','Jitter_local_abs','Jitter_rap','Jitter_ppq5','Jitter_ddp','Shimmer_local','Shimmer_local (db)','Shimmer_apq3','Shimmer_aqpq5','Shimmer_apq11','Shimmer_dda','hnr_mean']
    row= ["101.67","2.138","0.835","8.21097392201545e-05","0.441","0.5","1.323","6.727","0.583","3.544","4.283","5.79","10.631","16.67282975619477"]



    categories = ['Food Quality', 'Food Variety', 'Service Quality', 'Ambience', 'Affordability']
    categories = [*categories, categories[0]]

    restaurant_1 = [4, 4, 5, 4, 3]
    restaurant_2 = [5, 5, 4, 5, 2]
    restaurant_3 = [3, 4, 5, 3, 5]
    restaurant_1 = [*restaurant_1, restaurant_1[0]]
    restaurant_2 = [*restaurant_2, restaurant_2[0]]
    restaurant_3 = [*restaurant_3, restaurant_3[0]]


    fig = go.Figure(
        data=[
            go.Scatterpolar(r=restaurant_1, theta=categories, fill='toself', name='Restaurant 1'),
            go.Scatterpolar(r=restaurant_2, theta=categories, fill='toself', name='Restaurant 2'),
            go.Scatterpolar(r=restaurant_3, theta=categories, fill='toself', name='Restaurant 3')
        ],
        layout=go.Layout(
            title=go.layout.Title(text='Restaurant comparison'),
            polar={'radialaxis': {'visible': True}},
            showlegend=True
        )
    )

    pyo.plot(fig)
