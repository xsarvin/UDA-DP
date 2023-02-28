// ScriptServer.cpp : Implementation of CScriptServer
#include "stdafx.h"
#include "ScriptServer.h"

/////////////////////////////////////////////////////////////////////////////
// CScriptServer

STDMETHODIMP CScriptServer::ProcessScript(unsigned char* szScript)
{
	DWORD dwResult = WaitForSingleObject(m_hMutex, 30000);
	if(dwResult == WAIT_TIMEOUT)
		return E_FAIL;
	pushScript(szScript);
	HRESULT hr = SendScript();
	ReleaseMutex(m_hMutex);
	return hr;
}

STDMETHODIMP CScriptServer::FindServer(VARIANT_BOOL* pVFound)
{
	HRESULT hr = m_pConn->FindServer();
	*pVFound = (hr == S_OK) ? VARIANT_TRUE : VARIANT_FALSE;
	return hr;
}

STDMETHODIMP CScriptServer::GetServerPort(ULONG* pPort)
{
	if(SUCCEEDED(m_pConn->FindServer()))
		*pPort = (ULONG)m_pConn->GetPort();
	else *pPort = 0;
	return S_OK;
}

STDMETHODIMP CScriptServer::GetServerKey(ULONG* pKey)
{
	if(SUCCEEDED(m_pConn->FindServer()))
		*pKey = (ULONG)m_pConn->GetKey();
	else *pKey = 0;
	return S_OK;
}

STDMETHODIMP CScriptServer::SendScript()
{
	HRESULT hr = m_pConn->Connect();
	if(SUCCEEDED(hr))
	{
		Script_ *pScr = popScript();
		if(pScr != 0)
		{
			char* pStr = (char*)pScr->szScript;
			m_pConn->SendData(pStr, strlen(pStr));
			delete pScr;
		}
	}
	m_pConn->Disconnect();
	return hr;
}





