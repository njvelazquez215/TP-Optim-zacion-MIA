from implementacion import experimento, datainit

## General
data = datainit(plotFlag=False, modFlag=False)
num_epochs = 50
#########################################
## SGD
optimizador = 'SGD'
optimAux = None

bs = 10

## Fix
schedule = 'Fix'
schAux = None

step_size = 0.03

train_loss = experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, plotFlag=False)
print(f'Train loss para {optimizador} con LRS {schedule}: {train_loss}')

## Exponential
schedule = 'Exponential'
schAux = (400000,)

step_size = 0.05

train_loss = experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, plotFlag=False)
print(f'Train loss para {optimizador} con LRS {schedule}: {train_loss}')

## Power
schedule = 'Power'
schAux = (2e6, 0.75)

step_size = 0.05

train_loss = experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, plotFlag=False)
print(f'Train loss para {optimizador} con LRS {schedule}: {train_loss}')

## Performance
schedule = 'Performance'
schAux = (1.05, 0.6)

step_size = 0.015

train_loss = experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, plotFlag=False)
print(f'Train loss para {optimizador} con LRS {schedule}: {train_loss}')
#########################################
## ADAM
optimizador = 'ADAM'

## Fix
schedule = 'Fix'
schAux = None

optimAux = (0.95, 0.94)

bs = 120
step_size = 0.005
train_loss = experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, plotFlag=False)
print(f'Train loss para {optimizador} con LRS {schedule}: {train_loss}')

## Exponential
schedule = 'Exponential'
schAux = (500000,)

optimAux = (0.9, 0.95)

bs = 100
step_size = 0.002

train_loss = experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, plotFlag=False)
print(f'Train loss para {optimizador} con LRS {schedule}: {train_loss}')

## Power
schedule = 'Power'
schAux = (30000, 0.75)

optimAux = (0.9, 0.99)

bs = 120
step_size = 0.03

train_loss = experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, plotFlag=False)
print(f'Train loss para {optimizador} con LRS {schedule}: {train_loss}')

## Performance
schedule = 'Performance'
schAux = (1.05, 0.25)

optimAux = (0.6, 0.98)

bs = 60
step_size = 0.002

train_loss = experimento(data, optimizador, schedule, num_epochs, step_size, bs, optimAux, schAux, plotFlag=False)
print(f'Train loss para {optimizador} con LRS {schedule}: {train_loss}')


