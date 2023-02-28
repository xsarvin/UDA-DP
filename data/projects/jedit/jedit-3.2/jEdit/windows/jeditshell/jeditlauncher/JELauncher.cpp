// JELauncher.cpp : Implementation of CJEditLauncher

#include "stdafx.h"
#include "Jeditlauncher.h"
#include "FileList.h"
#include "ScriptWriter.h"
#include "JELauncher.h"
#include <assert.h>

/////////////////////////////////////////////////////////////////////////////
// CJEditLauncher

CJEditLauncher::CJEditLauncher()
	: m_bRunDiff(FALSE), m_pFileList(0),
	  m_pScriptServer(0), m_hJEditProcess(0),
	  m_bSendScriptOnLaunch(FALSE),
	  m_nIDTimer(0), m_nCounter(0),
	  m_nDelayedRelease(0)
{
	m_pScriptServer = new CScriptServer();
	_Module.pLauncher = this;
#if defined SPECIAL_BUILD
	hFile = 0;
#endif
}


CJEditLauncher::~CJEditLauncher()
{
	_Module.pLauncher = 0;
	delete m_pFileList;
	delete m_pScriptServer;
#if defined SPECIAL_BUILD
	if(hFile)
	{
		CloseHandle(hFile);
		hFile = 0;
	}
#endif
}

STDMETHODIMP CJEditLauncher::InterfaceSupportsErrorInfo(REFIID riid)
{
	static const IID* arr[] =
	{
		&IID_IJEditLauncher
	};
	for (int i=0; i < sizeof(arr) / sizeof(arr[0]); i++)
	{
		if (::ATL::InlineIsEqualGUID(*arr[i],riid))
			return S_OK;
	}
	return S_FALSE;
}

/////////////////////////////////////////////////////////////////////////////
//
// IJEditLauncher implementation

// public interface functions
// the get functions will set the out parameter to 0
// if the server file cannot be located

STDMETHODIMP CJEditLauncher::get_ServerKey(ULONG * pKey)
{
	return m_pScriptServer->GetServerKey(pKey);
}


STDMETHODIMP CJEditLauncher::get_ServerPort(ULONG * pPort)
{
	return m_pScriptServer->GetServerPort(pPort);
}

STDMETHODIMP CJEditLauncher::RunScript(BSTR bstrFileName)
{
	wchar_t* pwszPath = (wchar_t*)bstrFileName;
	return RunScript_WChar(pwszPath);
}

STDMETHODIMP CJEditLauncher::RunScript_WChar(wchar_t* pwszFileName)
{
	if(pwszFileName == 0)
		return E_FAIL;
	delete m_pFileList;
	m_pFileList = 0;
	m_pFileList = new BeanShellFileList(pwszFileName, true);
	return FindTarget(TRUE);
}

STDMETHODIMP CJEditLauncher::RunScript_Char(char* pszFileName)
{
	if(pszFileName == 0)
		return E_FAIL;
	delete m_pFileList;
	m_pFileList = 0;
	m_pFileList = new BeanShellFileList(pszFileName, true);
	return FindTarget(TRUE);
}

STDMETHODIMP CJEditLauncher::EvalScript(BSTR bstrFileName)
{
	wchar_t* pwszPath = (wchar_t*)bstrFileName;
	return EvalScript_WChar(pwszPath);
}

STDMETHODIMP CJEditLauncher::EvalScript_Char(char* pszScript)
{
	if(pszScript == 0)
		return E_FAIL;
	delete m_pFileList;
	m_pFileList = 0;
	m_pFileList = new BeanShellFileList(pszScript, true);
	return FindTarget(TRUE);
};


STDMETHODIMP CJEditLauncher::EvalScript_WChar(wchar_t* pwszScript)
{
	if(pwszScript == 0)
		return E_FAIL;
	delete m_pFileList;
	m_pFileList = 0;
	m_pFileList = new BeanShellFileList(pwszScript, false);
	return FindTarget(TRUE);
}

STDMETHODIMP CJEditLauncher::RunDiff_Var(VARIANTARG varFileNames)
{
	delete m_pFileList;
	m_pFileList = 0;
	ScriptWriter *pWriter = new OpenDiffScript();
	m_pFileList = new VariantFileList(pWriter, varFileNames);
	return FindTarget(TRUE);
}

STDMETHODIMP CJEditLauncher::RunDiff(BSTR bstrFileBase,
	BSTR bstrFileChanged)
{
	wchar_t *pwszFileBase = (wchar_t*)bstrFileBase;
	wchar_t *pwszFileChanged = (wchar_t*)bstrFileChanged;
	return RunDiff_WChar(pwszFileBase, pwszFileChanged);
}

STDMETHODIMP CJEditLauncher::RunDiff_WChar(wchar_t* pwszFileBase,
			wchar_t* pwszFileChanged)
{
	if(pwszFileBase == 0 || pwszFileChanged == 0)
		return E_FAIL;
	delete m_pFileList;
	m_pFileList = 0;
	ScriptWriter *pWriter = new OpenDiffScript();
	m_pFileList = new WideFilePair(pWriter, pwszFileBase, pwszFileChanged);
	return FindTarget(TRUE);
}

STDMETHODIMP CJEditLauncher::RunDiff_Char(char* pszFileBase,
	char* pszFileChanged)
{
	if(pszFileBase == 0 || pszFileChanged == 0)
		return E_FAIL;
	delete m_pFileList;
	m_pFileList = 0;
	ScriptWriter *pWriter = new OpenDiffScript();
	m_pFileList = new SimpleFilePair(pWriter, pszFileBase, pszFileChanged);
	return FindTarget(TRUE);
}

STDMETHODIMP CJEditLauncher::OpenFile(BSTR bstrFileName)
{
#if defined SPECIAL_BUILD
	WriteLogFile("Calling OpenFile() with wide char parameter.\n");
#endif
	wchar_t* pwszPath = (wchar_t*)bstrFileName;
	return OpenFile_WChar(pwszPath);
}

STDMETHODIMP CJEditLauncher::FindTarget(BOOL bSendScript)
{
	m_bSendScriptOnLaunch = bSendScript;
	HRESULT hr = S_OK;
	VARIANT_BOOL bVarFound;
	m_pScriptServer->FindServer(&bVarFound);
	if(bVarFound == VARIANT_FALSE)
	{
		hr = Launch();
		if(hr == S_OK)
		{
			m_bSendScriptOnLaunch = bSendScript;
			m_nCounter = 40;
			m_nIDTimer = ::SetTimer(0, 0, 500, LaunchTimerProc);
		}
	}
	else if(m_bSendScriptOnLaunch)
	{
		hr = m_pFileList->Process(m_pScriptServer);
		delete m_pFileList;
		m_pFileList = 0;
	}
	return hr;
}

// version 3.2 uses OpenFileScript

STDMETHODIMP CJEditLauncher::OpenFiles(VARIANTARG fileNames)
{
#if defined SPECIAL_BUILD
	WriteLogFile("[launcher] Calling OpenFiles() passing VARIANT parameter.\n");
#endif
	delete m_pFileList;
	m_pFileList = 0;
	ScriptWriter *pWriter = new OpenFileScript;
	m_pFileList = new VariantFileList(pWriter, fileNames);
	return FindTarget(TRUE);
}

STDMETHODIMP CJEditLauncher::Launch()
{
	if(m_hJEditProcess != 0)
	{
		DWORD dwResult = WaitForSingleObject(m_hJEditProcess, 0);
		if(dwResult == WAIT_TIMEOUT)
			return S_OK;
		else m_hJEditProcess = 0;
	}
	char* pScript = 0;
	char* pDummy = 0;
	StartAppScript script;
	script.WriteScript(&pDummy, 0, &pScript);
	return Launch_jEdit(pScript);
}

STDMETHODIMP CJEditLauncher::OpenFile_Char(CHAR* szFileName)
{
#if defined SPECIAL_BUILD
	WriteLogFile("[launcher] Calling OpenFile_Char() passing char parameter: ");
	WriteLogFile(szFileName);
	WriteLogFile("\n");
#endif
	return OpenFiles_Char(&szFileName, 1);
}

STDMETHODIMP CJEditLauncher::OpenFile_WChar(WCHAR* wszFileName)
{
#if defined SPECIAL_BUILD
	WriteLogFile("[launcher] Calling OpenFile_WChar().\n");
#endif
	return OpenFiles_WChar(&wszFileName, 1);
}


STDMETHODIMP CJEditLauncher::OpenFiles_Char(char **argv, int numArgs)
{
#if defined SPECIAL_BUILD
	WriteLogFile("[launcher] Calling OpenFiles_Char() with ");
	char szNum[32];
	itoa(numArgs, szNum, 10);
	strcat(szNum, " arguments\n");
	WriteLogFile(szNum);
#endif
	delete m_pFileList;
	m_pFileList = 0;
	ScriptWriter *pWriter = new OpenFileScript;
	m_pFileList = new SimpleFileList(pWriter, argv, numArgs);
	return FindTarget(TRUE);
}

STDMETHODIMP CJEditLauncher::OpenFiles_WChar(wchar_t **argv, int numArgs)
{
#if defined SPECIAL_BUILD
	WriteLogFile("[launcher] Calling OpenFiles_WChar() with ");
	char szNum[32];
	itoa(numArgs, szNum, 10);
	strcat(szNum, " arguments\n");
	WriteLogFile(szNum);
#endif
	delete m_pFileList;
	m_pFileList = 0;
	ScriptWriter *pWriter = new OpenFileScript;
	m_pFileList = new WideFileList(pWriter, argv, numArgs);
	return FindTarget(TRUE);
}


HRESULT CJEditLauncher::Launch_jEdit(char* szCmdLine)
{
#if defined SPECIAL_BUILD
	WriteLogFile("[launcher] Calling Launch_jEdit()\n[launcher] Command line: ");
	WriteLogFile(szCmdLine);
	WriteLogFile("\n");
#endif
	HKEY hKey;
	LONG nResult;
	TCHAR szTemp[MAX_PATH],
		  szKeyPath[MAX_PATH];
	const TCHAR space[2] = {_T(' '), 0};
	DWORD dwCount = MAX_PATH * sizeof(TCHAR);
	DWORD dwType = 0;
	LoadString(_Module.GetModuleInstance(), IDS_REG_PARAMS_KEY_3_2,
		szKeyPath, MAX_PATH);
	nResult = RegOpenKeyEx(HKEY_CURRENT_USER, szKeyPath, 0, KEY_READ, &hKey);
	if(nResult != ERROR_SUCCESS)
	{
		MakeErrorInfo(IDS_ERR_NO_REGISTRY_KEY);
	}
	else
	{
		dwCount = MAX_PATH;
		nResult = RegQueryValueEx(hKey, _T("jEdit Working Directory"), 0, &dwType, (LPBYTE)szTemp, &dwCount);
		if(nResult != ERROR_SUCCESS)
		{
			MakeErrorInfo(IDS_ERR_NO_JEDIT_WORKINGDIR_VALUE);
		}
	}

	RegCloseKey(hKey);
	if(nResult != ERROR_SUCCESS)
		return E_FAIL;

	STARTUPINFO si;
	::ZeroMemory(&si, sizeof(si));
	PROCESS_INFORMATION pi;
	BOOL bReturn = CreateProcess(0, szCmdLine, 0, 0, 0, 0, 0, szTemp, &si, &pi);
	if(!bReturn)
	{
		LPSTR szErrMsg;
		::FormatMessageA(
			FORMAT_MESSAGE_ALLOCATE_BUFFER |
			FORMAT_MESSAGE_FROM_SYSTEM |
			FORMAT_MESSAGE_IGNORE_INSERTS,
			NULL,
			GetLastError(),
			MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
			(LPSTR) &szErrMsg,
			0,
			NULL
		);
		MakeErrorInfo(szErrMsg);
		::LocalFree((LPVOID)szErrMsg);
	}
	else
		m_hJEditProcess = pi.hProcess;
	return bReturn ? S_OK : E_FAIL;
}


// provides error string from string table
// resource to be returned to script engine;
// available from C or C++ client code by calling ::GetErrorInfo()

void CJEditLauncher::MakeErrorInfo(UINT nErrorStringID)
{
	if(nErrorStringID == 0)
	{
		::SetErrorInfo(0L, 0);
		return;
	}
	CHAR errorMsg[256];
	HINSTANCE hInstance = _Module.GetModuleInstance();
	LoadString(hInstance, nErrorStringID, errorMsg, 255);

	if(*errorMsg == 0)
		LoadString(hInstance, IDS_ERR_UNSPECIFIED, errorMsg, 255);
	MakeErrorInfo(errorMsg);
}

void CJEditLauncher::MakeErrorInfo(CHAR* pszErrorMsg)
{
	if(pszErrorMsg == 0)
		return;
	int len = strlen(pszErrorMsg);
	WCHAR pwszErrorMsg[256];
	ZeroMemory(pwszErrorMsg, sizeof(WCHAR) * 256);
	MultiByteToWideChar(CP_ACP, 0, pszErrorMsg, len, pwszErrorMsg, len);
	ICreateErrorInfo *piCreateErr = 0;
	HRESULT hr = ::CreateErrorInfo( &piCreateErr);
	if(FAILED(hr))
		return;
	//piCreateErr->SetHelpFile(...);
	//piCreateErr->SetHelpContext(...);
	piCreateErr->SetSource(L"JEdit.JEditLauncher");
	piCreateErr->SetDescription(pwszErrorMsg);

	IErrorInfo *piError = 0;
	hr = piCreateErr->QueryInterface(IID_IErrorInfo, (void**)&piError);
	if(SUCCEEDED(hr))
	{
		::SetErrorInfo(0L, piError);
		piError->Release();
	}
	piCreateErr->Release();
}

void CJEditLauncher::OnTimer(UINT nIDTimer)
{
	if(nIDTimer != m_nIDTimer) return;
	VARIANT_BOOL bVarFound;
	m_pScriptServer->FindServer(&bVarFound);
	if(bVarFound == VARIANT_TRUE || m_nCounter == 0)
	{
		::OutputDebugString("Killing timer\n");
		BOOL bKilled = KillTimer(0, m_nIDTimer);
		::OutputDebugString(bKilled ? "Timer killed\n" : "Timer not killed\n");
		m_nIDTimer = 0;
		if(bVarFound == VARIANT_TRUE && m_bSendScriptOnLaunch)
		{
			assert(m_pFileList);
			assert(m_pScriptServer);
			m_pFileList->Process(m_pScriptServer);
			delete m_pFileList;
			m_pFileList = 0;
			::OutputDebugString("File List processed\n");
		}
		OutputDebugString("Delayed Release\n");
		while(m_nDelayedRelease-- > 0)
		{
			_Module.Unlock();
		}
	}
	else --m_nCounter;
}

/*
ULONG CJEditLauncher::Release()
{
	// do not release if a timer is running
	if(m_nIDTimer != 0)
	{
		MessageBox(0, "Release not done", "jEditLauncher", 0);
		++m_nDelayedRelease;
	}
	else
		InternalRelease();
	return 0;  // forget about debug version return;
}
*/


void CALLBACK LaunchTimerProc(HWND hwnd, UINT uMsg,
	UINT_PTR idEvent, DWORD dwTime)
{
	//::MessageBox(0, "Timer fired", "jEditLauncher", 0);
	_Module.pLauncher->OnTimer(idEvent);
}

// special build function

#if defined SPECIAL_BUILD
void CJEditLauncher::OpenLogFile()
{
	if(hFile != 0)
		return;
	// special build debug code
	char szFile[MAX_PATH];
	GetModuleFileName(NULL, szFile, MAX_PATH);
	char *pSlash = strrchr(szFile, '\\');
	strcpy(pSlash + 1, "jedebug.log");

	hFile = CreateFile(szFile, GENERIC_WRITE, 
		FILE_SHARE_READ | FILE_SHARE_WRITE, 
		0, OPEN_ALWAYS, FILE_ATTRIBUTE_ARCHIVE, 0);
	SetFilePointer(hFile, 0, 0, SEEK_END);
}

void CJEditLauncher::WriteLogFile(const char* szMsg)
{
	if(hFile == 0)
		OpenLogFile();
	DWORD dwLen;
	WriteFile(hFile, szMsg, strlen(szMsg), &dwLen, 0);
}

void CJEditLauncher::CloseLogFile()
{
	if(hFile != 0)
	{
		CloseHandle(hFile);
		hFile = 0;
	}
}
#endif

