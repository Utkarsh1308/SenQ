from django.urls import path
from . import views

app_name = 'graph'

urlpatterns = [
    # /graph/
    path('', views.index, name='index'),

    # /graph/register/
    path('register', views.register, name='register'),

    # /graph/login/
    path('login', views.login_user, name='login'),

    # /graph/logout
    path('logout_user', views.logout_user, name='logout_user'),

    # /graph/plot/
    path('plot', views.plot, name='plot'),

    # /graph/shares/
    path('shares', views.advanced, name='advanced'),

    # /graph/pnl/
    path('pnl', views.pnl, name='pnl'),

    # /graph/backtest/
    path('backtest', views.backtest, name='backtest'),

    # /graph/backtest2/
    path('backtest2', views.backtest2, name='backtest2'),
]
