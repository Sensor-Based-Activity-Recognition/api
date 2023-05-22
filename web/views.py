# from django.shortcuts import render
import io
import json
import brotli
from web.utils.pipelineCNN import PipelineCNN
from web.utils.modelCNN import Runner as RunnerCNN
import pandas as pd

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt


# Create your views here.
@csrf_exempt
def CNN(request):
    # get the request body
    body = io.BytesIO(request.body)
    # decode the body using brotli
    buffer = io.BytesIO(brotli.decompress(body.read()))
    # convert the parquet body to a dataframe
    data = pd.read_parquet(buffer)

    # run through pipeline
    data = PipelineCNN().run(data)
    # run through model
    runner = RunnerCNN("web/utils/modelCNN.pkl")
    result = runner.run(data)

    # dictionary to json
    result = json.dumps(result)

    # return the result
    return HttpResponse(result)
