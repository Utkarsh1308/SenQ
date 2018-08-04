from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.contrib.auth import authenticate, login, logout
from django.views import generic
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from django.views.generic import View
from random import randrange

from .forms import ContactForm, UserForm
from django.contrib.auth.models import User

from graph.backtest import *


# Create your views here.
def index(request):
    template = loader.get_template('graph/index.html')
    context = {}
    return HttpResponse(template.render(context, request))

def plot(request):
    context = {}
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            global values2, values3, header2, header3, data_for_x, data_for_y
            alpha = form.cleaned_data['alpha']
            [data_for_x, data_for_y, values, values2, values3, header2, header3] = plot_graph(str(alpha))
            data_for_x = [str(date) for date in data_for_x[0:len(data_for_y)]]
            return render(request, 'graph/index.html', {
            'form': form,
            'alpha': alpha,
            'data_for_x': data_for_x,
            'data_for_y': data_for_y,
            'values': values,
            'values2': values2,
            'values3': values3,
            })

def backtest(request):
    template = loader.get_template('graph/backtest.html')

    textarea ="""
    Enter long position on 2016-07-11 Quantity: 40 at 458.0 \n
    Exit long position on 2016-07-22 Quantity: 40 at 454.8 PNL: -128.0 in 9 days\n
    Enter long position on 2016-07-25 Quantity: 40 at 464.6\n
    Exit long position on 2016-10-17 Quantity: 40 at 531.0 PNL: 2656.0 in 55 days\n
    Exit short position on 2016-11-15 Quantity: 12 at 483.0 PNL: -1189.8 in 178 days\n
    Enter short position on 2016-11-29 Quantity: 12 at 454.0\n
    Enter long position on 2017-01-02 Quantity: 40 at 453.05\n
    Exit long position on 2017-02-28 Quantity: 40 at 504.6 PNL: 2062.0 in 39 days\n
    Enter long position on 2017-03-01 Quantity: 40 at 510.0\n
    Exit long position on 2017-03-03 Quantity: 40 at 500.0 PNL: -400.0 in 2 days\n
    Enter long position on 2017-03-07 Quantity: 40 at 509.0\n
    Exit long position on 2017-04-17 Quantity: 40 at 517.3 PNL: 332.0 in 26 days\n
    Enter long position on 2017-04-21 Quantity: 40 at 536.85\n
    Exit long position on 2017-04-25 Quantity: 40 at 526.5 PNL: -414.0 in 2 days\n
    Enter long position on 2017-04-26 Quantity: 40 at 528.0\n
    Exit long position on 2017-05-02 Quantity: 40 at 527.5 PNL: -20.0 in 3 days\n
    Enter long position on 2017-05-03 Quantity: 40 at 536.0\n
    Exit long position on 2017-05-04 Quantity: 40 at 530.0 PNL: -240.0 in 1 days\n
    Exit short position on 2017-05-10 Quantity: 12 at 496.0 PNL: -504.0 in 110 days\n
    Enter long position on 2017-05-12 Quantity: 40 at 543.7\n
    Enter short position on 2017-05-12 Quantity: 12 at 543.7\n
    Exit long position on 2017-05-17 Quantity: 40 at 522.6 PNL: -844.0 in 3 days\n
    Enter long position on 2017-05-18 Quantity: 40 at 517.0\n
    Exit long position on 2017-05-19 Quantity: 40 at 517.0 PNL: 0.0 in 1 days\n
    Enter long position on 2017-05-30 Quantity: 40 at 521.8\n
    Exit long position on 2017-06-01 Quantity: 40 at 515.6 PNL: -248.0 in 2 days\n
    Enter long position on 2017-06-02 Quantity: 40 at 527.0\n
    Exit long position on 2017-06-07 Quantity: 40 at 519.0 PNL: -320.0 in 3 days\n
    Enter long position on 2017-06-09 Quantity: 40 at 521.5\n
    Exit long position on 2017-06-13 Quantity: 40 at 517.5 PNL: -160.0 in 2 days\n
    Exit short position on 2017-06-28 Quantity: 12 at 490.0 PNL: 644.4 in 32 days\n
    Enter short position on 2017-07-06 Quantity: 12 at 502.0\n
    Enter long position on 2017-07-10 Quantity: 40 at 494.7\n
    Exit long position on 2017-08-11 Quantity: 40 at 517.0 PNL: 892.0 in 24 days\n
    Exit short position on 2017-08-14 Quantity: 12 at 509.2 PNL: -86.4 in 27 days\n
    Enter short position on 2017-08-29 Quantity: 12 at 518.4\n
    Enter long position on 2017-09-04 Quantity: 40 at 523.0\n
    Exit long position on 2017-09-05 Quantity: 40 at 522.0 PNL: -40.0 in 1 days\n
    Enter long position on 2017-09-13 Quantity: 40 at 534.1\n
    Exit long position on 2017-09-25 Quantity: 40 at 520.25 PNL: -554.0 in 8 days\n
    Exit short position on 2017-10-18 Quantity: 12 at 490.1 PNL: 339.6 in 35 days\n
    """

    context = {}
    return HttpResponse(template.render(context, request))

def backtest2(request):
    Alpha1 = 'mean_sma(-(delta(c,5)/delay(c,5)-cs_mean(delta(c,5)/delay(c,5))),5)'
    Alpha2 = 'c'
    Alpha3 = '-c'
    Alpha4 = 'o-c'
    Alpha5 = 'c-o'
    alphas = [Alpha1, Alpha2, Alpha3, Alpha4, Alpha5]
    random_index = randrange(0,len(alphas))
    [data_for_x, data_for_y, values, values2, values3, header2, header3] = plot_graph(str(alphas[random_index]))
    template = loader.get_template('graph/backtest2.html')
    data_for_x = [str(date) for date in data_for_x[0:len(data_for_y)]]
    context = {
    'data_for_x': data_for_x,
    'data_for_y': data_for_y,
    }
    return HttpResponse(template.render(context, request))

def advanced(request):
    return render(request, 'graph/shares.html', {
    'values2': values2,
    'header2': header2,
    })

def pnl(request):
    return render(request, 'graph/pnl.html', {
    'values3': values3,
    'header3': header3,
    })


def register(request):
    form = UserForm(request.POST or None)
    if form.is_valid():
        user = form.save(commit=False)
        username = form.cleaned_data['username']
        password = form.cleaned_data['password']
        user.set_password(password)
        user.save()
        user = authenticate(username=username, password=password)
        if user is not None:
            if user.is_active:
                login(request, user)
                return redirect('graph:index')
    context = {
        "form": form,
    }
    return render(request, 'graph/register.html', context)

def login_user(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is not None:
            if user.is_active:
                login(request, user)
                user = User.objects.get(username=username)
                return render(request, 'graph/index.html', {'user': user})
            else:
                return render(request, 'graph/login.html', {'error_message': 'Your account has been disabled'})
        else:
            return render(request, 'graph/login.html', {'error_message': 'Invalid login'})
    return render(request, 'graph/login.html')

def logout_user(request):
    logout(request)
    form = UserForm(request.POST or None)
    context = {
        "form": form,
    }
    return render(request, 'graph/index.html', context)
