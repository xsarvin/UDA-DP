// ScriptServer.h : Declaration of the CScriptServer

#ifndef __SCRIPTSERVER_H_
#define __SCRIPTSERVER_H_

#include "resource.h"       // main symbols
#include "ServConn.h"


// interface IScriptServer

interface IScriptServer
{
	virtual HRESULT STDMETHODCALLTYPE ProcessScript
		(unsigned char *szScript) = 0;
	virtual HRESULT STDMETHODCALLTYPE FindServer
		(VARIANT_BOOL* pVFound) = 0;
	virtual HRESULT STDMETHODCALLTYPE GetServerPort
		(ULONG* pPort) = 0;
	virtual HRESULT STDMETHODCALLTYPE GetServerKey
		(ULONG* pKey) = 0;
};

class CScriptServer : public IScriptServer
{
public:
	CScriptServer()
		: firstScript(0), lastScript(0), m_hMutex(0)
	{
		m_hMutex = CreateMutex(0, FALSE, _T("jEditLauncher"));
		m_pConn = new ServerConnection();
	}
	~CScriptServer()
	{
		ReleaseMutex(m_hMutex);
		delete m_pConn;
		while(Script_* pScr = popScript()) delete pScr;
	}

public:
	virtual HRESULT STDMETHODCALLTYPE ProcessScript(unsigned char *szScript);
	virtual HRESULT STDMETHODCALLTYPE FindServer(VARIANT_BOOL* pVFound);
	virtual HRESULT STDMETHODCALLTYPE GetServerPort(ULONG* pPort);
	virtual HRESULT STDMETHODCALLTYPE GetServerKey(ULONG* pKey);
protected:
	struct Script_ {
		unsigned char *szScript;
		Script_ *next;
		Script_(unsigned char* szBuf)
			: szScript(0), next(0)
		{
			int len = strlen((const char*)szBuf) + 1;
			szScript = (unsigned char*)CoTaskMemAlloc(len);
			//ZeroMemory(szScript, len);
			CopyMemory(szScript, szBuf, len);
			//MessageBox(0, (char*)szScript, "new buf inside Script_", 0);
		}
		~Script_() { CoTaskMemFree(szScript); }
	};
	Script_ *firstScript, *lastScript;
	void pushScript(unsigned char *szBuf)
	{
//		MessageBox(0, (char*)szBuf, "new buf for script", 0);
		Script_ *pScr = new Script_(szBuf);
		if(firstScript == 0)
			firstScript = pScr;
		else
			lastScript->next = pScr;
		lastScript = pScr;
	}
	Script_* popScript()
	{
		Script_* pScr = firstScript;
		if(pScr != 0)
			firstScript = pScr->next;
		if(lastScript == pScr)
			lastScript = 0;
		return pScr;
	}
	HRESULT STDMETHODCALLTYPE SendScript();

private:
	HANDLE m_hMutex;
	ServerConnection *m_pConn;
};


#endif //__SCRIPTSERVER_H_
