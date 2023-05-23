# from django.shortcuts import render
import io
import json
import gzip
from web.utils.pipeline import Pipeline
from web.utils.modelCNN import Runner as RunnerCNN
from web.utils.modelHGBC import Runner as RunnerHGBC
import pandas as pd

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def index(request):
    return HttpResponse("Use /CNN or /HGBC for API Requests")


# Create your views here.
@csrf_exempt
def CNN(request):
    # decompress the request body
    body = gzip.decompress(request.body)
    # decode the body using utf-8
    body = io.StringIO(body.decode("utf-8"))
    # convert the csv body to a dataframe
    data = pd.read_csv(body)

    # run through pipeline
    data = Pipeline().run(data, model="CNN")
    # run through model
    runner = RunnerCNN("web/utils/modelCNN.pkl")
    result = runner.run(data)

    # dictionary to json
    result = json.dumps(result)

    # return the result
    return HttpResponse(result)


@csrf_exempt
def HGBC(request):
    # decompress the request body
    body = gzip.decompress(request.body)
    # decode the body using utf-8
    body = io.StringIO(body.decode("utf-8"))
    # convert the csv body to a dataframe
    data = pd.read_csv(body)

    # run through pipeline
    data = Pipeline().run(data, model="HGBC")
    # run through model
    runner = RunnerHGBC("web/utils/modelHGBC.pkl")
    result = runner.run(data)

    # dictionary to json
    result = json.dumps(result)

    # return the result
    return HttpResponse(result)
