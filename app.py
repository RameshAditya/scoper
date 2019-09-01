"""
Flask server to run Scoper with a GUI

"""

from flask import Flask, render_template, request
import nltk
from scoper import Scoper

nltk.download("brown")

APP = Flask(__name__)

print("Please wait loading instance.....")
FINDER = Scoper()


@APP.route("/", methods=["GET", "POST"])
def index():
    """
    Home page of Scoper. 

    """
    if request.method == "POST":
        youtube_link = request.form["youtube_link"]
        algorithm = request.form["algorithm"]
        limit = request.form["limit"]
        query = request.form["query"]
        print(query)

        # check if the values are present
        if not (youtube_link and algorithm and limit):
            return render_template(
                "index.html", error="please enter proper values in the form", empty=True
            )

        # get the query results
        results = FINDER.main(
            youtube_link,
            query=query,
            limit=int(limit),
            languages=["en"],
            mode=algorithm,
        )

        # converting results from [('apple watch.', '29m 56s')] to [['apple watch.', '29m56s']]
        results = [[x[0], x[1].replace(" ", "")] for x in results]

        print(results)
        return render_template("index.html", videolink=youtube_link, results=results)

    return render_template("index.html", empty=True)


if __name__ == "__main__":
    APP.run()
