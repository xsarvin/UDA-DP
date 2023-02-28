// JELauncher.h : Declaration of the CJEditLauncher

#ifndef __JEDITLAUNCHER_H_
#define __JEDITLAUNCHER_H_

#include "jeditlauncher.h"
#include "ServConn.h"
#include "ScriptServer.h"

class FileListImpl;
/////////////////////////////////////////////////////////////////////////////
// CJEditLauncher
class ATL_NO_VTABLE CJEditLauncher :
	public CComObjectRootEx<CComSingleThreadModel>,
	public CComCoClass<CJEditLauncher, &CLSID_JEditLauncher32>,
	public ISupportErrorInfo,
	public IDispatchImpl<IJEditLauncher, &IID_IJEditLauncher, &LIBID_JEDITLAUNCHERLib, 3, 2>
{
public:
	DECLARE_CLASSFACTORY_SINGLETON(CJEditLauncher)
	CJEditLauncher();
	~CJEditLauncher();

DECLARE_REGISTRY_RESOURCEID(IDR_JEDITLAUNCHER32)

DECLARE_PROTECT_FINAL_CONSTRUCT()

BEGIN_COM_MAP(CJEditLauncher)
	COM_INTERFACE_ENTRY(IJEditLauncher)
	COM_INTERFACE_ENTRY(IDispatch)
	COM_INTERFACE_ENTRY(ISupportErrorInfo)
END_COM_MAP()

// ISupportsErrorInfo
	STDMETHOD(InterfaceSupportsErrorInfo)(REFIID riid);

// IJEditLauncher interface

// Methods for dispatch interface
public:
    virtual HRESULT STDMETHODCALLTYPE get_ServerKey(ULONG * pKey);
    virtual HRESULT STDMETHODCALLTYPE get_ServerPort(ULONG * pPort);

    virtual HRESULT STDMETHODCALLTYPE OpenFile(BSTR bstrFileName);
    virtual HRESULT STDMETHODCALLTYPE OpenFiles(VARIANTARG fileNames);

    virtual HRESULT STDMETHODCALLTYPE Launch();

    virtual HRESULT STDMETHODCALLTYPE RunScript(BSTR bstrFileName);

    virtual HRESULT STDMETHODCALLTYPE EvalScript(BSTR bstrScript);

	virtual HRESULT STDMETHODCALLTYPE RunDiff(BSTR bstrFileBase,
						BSTR bstrFileRevised);

// for in-process server, low level functions can be called,
// but with change to out-of-prcess server, custom marshalling
// is necessary for non-automation data types
protected:
    virtual HRESULT STDMETHODCALLTYPE OpenFile_Char(char* szFileName);
    virtual HRESULT STDMETHODCALLTYPE OpenFile_WChar(wchar_t* wszFileName);
    virtual HRESULT STDMETHODCALLTYPE OpenFiles_Char(char** argv, int numArgs);
    virtual HRESULT STDMETHODCALLTYPE OpenFiles_WChar(wchar_t** argv, int numArgs);
    virtual HRESULT STDMETHODCALLTYPE RunScript_Char(char* pszFileName);
    virtual HRESULT STDMETHODCALLTYPE RunScript_WChar(wchar_t* pwszFileName);
    virtual HRESULT STDMETHODCALLTYPE EvalScript_Char(char* pszScript);
    virtual HRESULT STDMETHODCALLTYPE EvalScript_WChar(wchar_t* pwszScript);
	virtual HRESULT STDMETHODCALLTYPE RunDiff_Char(char* pszFileBase,
			char* pszFileChanged);
	virtual HRESULT STDMETHODCALLTYPE RunDiff_WChar(wchar_t* pwszFileBase,
			wchar_t* pwszFileChanged);
	virtual HRESULT STDMETHODCALLTYPE RunDiff_Var(VARIANTARG varFileNames);

    // public helper functions
public:
    static void MakeErrorInfo(UINT nErrorStringID);
    static void MakeErrorInfo(CHAR* pszErrorMsg);

	void OnTimer(UINT nIDTimer);
	BOOL IsTimer()
	{
		if(m_nIDTimer == 0)
			return FALSE;
		++m_nDelayedRelease;
		return TRUE;
	}

	// implementation
protected:
    HRESULT STDMETHODCALLTYPE FindTarget(BOOL bSendScript);
    HRESULT Launch_jEdit(char* szCmdLine);

private:
	BOOL m_bRunDiff;
    FileListImpl *m_pFileList;
	CScriptServer *m_pScriptServer;
	HANDLE m_hJEditProcess;
	BOOL m_bSendScriptOnLaunch;
	UINT m_nIDTimer;
	UINT m_nCounter;
	UINT m_nDelayedRelease;

#if defined SPECIAL_BUILD
public:
	HANDLE hFile;
	void OpenLogFile();
	void WriteLogFile(const char* szMsg);	
	void CloseLogFile();
#endif
};


void CALLBACK LaunchTimerProc(HWND hwnd, UINT uMsg,
	UINT_PTR idEvent, DWORD dwTime);


#endif //__JEDITLAUNCHER_H_
