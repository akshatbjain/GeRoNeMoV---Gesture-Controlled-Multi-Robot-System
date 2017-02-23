#include "tserial.h"

class serial{

  private:
        // private attributes
		Tserial *com;
  public:


	serial() {
			
		 }
	
	bool startDevice(char *port,int speed)
	{
		com = new Tserial();
		if (com!=0)
		{
			if(com->connect(port, speed, spNONE))
				printf("Not Connected...\n");
			else
				printf("Connected..\n");
			return TRUE;
		}
		else
			return FALSE; 
	}

	void stopDevice()
	{
		com->disconnect();
        // ------------------
        delete com;
        com = 0;
	}

	void send_data(unsigned char data)
	{
	//	unsigned char data = 0;
	
		
		com->sendChar(data);
		//printf("%d",data);
		
	}
	char read_data()
	{
		char x;
		if (com->getChar() == 0x7e)
		{
			for (int m = 0; m < 19; m++)
			{
				x = com->getChar();
			}
		}
		x = com->getChar();
		printf("%d", x);
		return x;
	}
};