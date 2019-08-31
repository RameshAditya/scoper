from flask import Flask, render_template, request
from scoper import Scoper
import nltk

nltk.download("brown")

app = Flask(__name__)

print("Please wait loading instance.....")
FINDER = Scoper()

@app.route("/",methods=["GET","POST"])
def index():
    """
        POST
    """
    if request.method == 'POST':
        youtube_link = request.form['youtube_link']
        algorithm = request.form['algorithm']
        limit = request.form['limit']
        query = request.form['query']
        print(query)
        
        # check if the values are present
        if not (youtube_link and algorithm and limit):
            return render_template("index.html", error="please enter proper values in the form", empty=True)
                
        # get the query results
        results = FINDER.main(youtube_link, query=query, limit=5, languages=["en"], mode=algorithm)

        # converting results from [('apple watch.', '29m 56s')] to [['apple watch.', '29m56s']]
        print("results before", results)
        results = [[x[0],x[1].replace(" ","")] for x in results]
        
        print(results)
        return render_template("index.html", videolink=youtube_link, results=results)

    """
        GET
    """
    return render_template("index.html", empty=True)
if __name__ == "__main__":
    app.run(debug=True)