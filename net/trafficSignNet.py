# import packages
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import MaxPool2d
from torch.nn import LogSoftmax
from torch.nn import BatchNorm2d
from torch import flatten

class trafficSignNet(Module):
	def __init__(self,numChannels,numClasses):
		# Call constructor
		super(trafficSignNet,self).__init__()

		# initialize net
		self.conv1 = Conv2d(in_channels=numChannels,out_channels=8, \
			kernel_size=(5,5),padding="same")
		self.relu1 = ReLU()
		self.mp1 = MaxPool2d(kernel_size=(2,2))

		self.conv2 = Conv2d(in_channels=8,out_channels=16, \
			kernel_size=(3,3), padding="same")
		self.relu2 = ReLU()
		self.conv3 = Conv2d(in_channels=16, out_channels=16, \
			kernel_size=(3,3), padding="same")
		self.relu3 = ReLU()
		self.mp2 = MaxPool2d(kernel_size=(2,2))

		self.conv4 = Conv2d(in_channels=16,out_channels=32, \
			kernel_size=(3,3), padding="same")
		self.relu4 = ReLU()
		self.conv5 = Conv2d(in_channels=32, out_channels=32, \
			kernel_size=(3,3), padding="same")
		self.relu5 =ReLU()
		self.mp3 = MaxPool2d(kernel_size=(2,2))

		self.fc1 = Linear(in_features=3*3*32,out_features=128)
		self.relu6 = ReLU()

		self.fc2 = Linear(in_features=128,out_features=numClasses)
		self.sm = LogSoftmax(dim=1)

	def forward(self, x):
		# pass input through net
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.mp1(x)

		x = self.conv2(x)
		x = self.relu2(x)
		x = self.conv3(x)
		x = self.relu3(x)
		x = self.mp2(x)

		x = self.conv4(x)
		x = self.relu4(x)
		x = self.conv5(x)
		x = self.relu5(x)
		x = self.mp6(x)

		x = flatten(x, 1)
		x = self.fc1(x)
		x = self.relu6(x)
		
		x = self.fc2(x)
		output = self.sm(x)

		return output


		
