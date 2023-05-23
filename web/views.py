# from django.shortcuts import render
import io
import json
import gzip
from web.utils.pipelineCNN import PipelineCNN
from web.utils.modelCNN import Runner as RunnerCNN
import pandas as pd

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt


# Create your views here.
@csrf_exempt
def CNN(request):
    # decompress the request body
    body = gzip.decompress(request.body)
    # decode the body using utf-8
    body = io.StringIO(body.decode("utf-8"))
    # convert the csv body to a dataframe
    data = pd.read_csv(body)
    # convert timestamp to datetime
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # run through pipeline
    data = PipelineCNN().run(data)
    # run through model
    runner = RunnerCNN("web/utils/modelCNN.pkl")
    result = runner.run(data)

    # dictionary to json
    result = json.dumps(result)

    # return the result
    return HttpResponse(result)
