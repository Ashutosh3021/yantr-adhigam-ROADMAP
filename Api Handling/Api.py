import requests
import pandas as pd
import time

df = pd.DataFrame()

for i in range(1, 534):
    response = requests.get(
        "https://api.themoviedb.org/3/movie/top_rated?api_key=YOUR_NEW_KEY&language=en-US&page={}".format(i)
    )

    temp_df = pd.DataFrame(response.json()['results'])[
        ["id","title","release_date","overview","popularity","vote_average","vote_count"]
    ]

    df = pd.concat([df, temp_df], ignore_index=True)

    time.sleep(0.25)

df.head()