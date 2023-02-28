/*
 * ScriptWriter.h - part of jEditLauncher package
 * Copyright (C) 2001 John Gellene
 * jgellene@nyc.rr.com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or any later version.
 *
 * Notwithstanding the terms of the General Public License, the author grants
 * permission to compile and link object code generated by the compilation of
 * this program with object code and libraries that are not subject to the
 * GNU General Public License, provided that the executable output of such
 * compilation shall be distributed with source code on substantially the
 * same basis as the jEditLauncher package of which this program is a part.
 * By way of example, a distribution would satisfy this condition if it
 * included a working makefile for any freely available make utility that
 * runs on the Windows family of operating systems. This condition does not
 * require a licensee of this software to distribute any proprietary software
 * (including header files and libraries) that is licensed under terms
 * prohibiting redistribution to third parties.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 *
 * $Id: ScriptWriter.h,v 1.3 2001/08/25 00:31:02 jgellene Exp $
 */

#if !defined(__SCRIPTWRITER_H__)
#define __SCRIPTWRITER_H__

/**
*@author author
*/

class ScriptWriter
{
	/* constructor */
public:
	ScriptWriter();
	virtual ~ScriptWriter();

	/* helper operation */
	static void MakeFullPath(LPSTR path);


	/* Override */
public:
	HRESULT WriteScript(VARIANTARG var, char** ppScript);
	HRESULT WriteScript(wchar_t* wargv[], int nArgs, char **ppScript);
	HRESULT WriteScript(char* argv[], int nArgs, char **ppScript);

	/* Protected implementation interface */
protected:
	virtual HRESULT WritePrefix() = 0;
	virtual HRESULT ProcessSinglePath(const char* path) = 0;
	virtual HRESULT WriteSuffix() = 0;

	/* Implementation */
protected:
	HRESULT InitBuffer(size_t size);
	void ClearBuffer();
	void ReleaseBuffer();
	char* GetBuffer();
	HRESULT CheckBuffer(size_t sizeCheck, size_t sizeIncr);
	HRESULT ProcessPathArray(VARIANTARG var);
	HRESULT ProcessPath(wchar_t* pwszPath);
	HRESULT ProcessPath(char* pszPath);
	void AppendPath(const char* path);
	void Append(const char* source);

private:
	static HRESULT ResolveLink(char* path, char* outPath);

private:
	char *pBuffer;
	char *pPathBuffer;
	size_t bufferSize;

	/* No copying */
private:
	ScriptWriter(const ScriptWriter&);
	ScriptWriter& operator=(const ScriptWriter&);
};

/**
*@author author
*/

class OpenFileScript : public ScriptWriter
{
	/* constructor */
public:
	OpenFileScript();
	virtual ~OpenFileScript();

	/* Overrides */
protected:
	virtual HRESULT WritePrefix();
	virtual HRESULT ProcessSinglePath(const char* path);
	virtual HRESULT WriteSuffix();

	/* Data */
private:
	int m_nFiles;

	/* No copying */
private:
	OpenFileScript(const OpenFileScript&);
	OpenFileScript& operator=(const OpenFileScript&);
};

/**
*@author author
*/

class OpenDiffScript : public ScriptWriter
{
	/* constructor */
public:
	OpenDiffScript();
	virtual ~OpenDiffScript();

	/* Overrides */
protected:
	virtual HRESULT WritePrefix();
	virtual HRESULT ProcessSinglePath(const char* path);
	virtual HRESULT WriteSuffix();

	/* Data */
protected:
	bool secondFile;
	/* No copying */
private:
	OpenDiffScript(const OpenDiffScript&);
	OpenDiffScript& operator=(const OpenDiffScript&);
};


/**
*@author author
*/

class StartAppScript : public ScriptWriter
{
	/* constructor */
public:
	StartAppScript();
	virtual ~StartAppScript();

	/* Overrides */
protected:
	virtual HRESULT WritePrefix();
	virtual HRESULT ProcessSinglePath(const char* path);
	virtual HRESULT WriteSuffix();

	/* Data */
private:
	bool bFirstFile;

	/* No copying */
private:
	StartAppScript(const StartAppScript&);
	StartAppScript& operator=(const StartAppScript&);
};


/**
*@author author
*/
class FileListScript : public ScriptWriter
{
	/* constructor */
public:
	FileListScript();
	virtual ~FileListScript();

	/* Overrides */
protected:
	virtual HRESULT WritePrefix();
	virtual HRESULT ProcessSinglePath(const char* path);
	virtual HRESULT WriteSuffix();

	/* Data */
private:
	bool bFirstFile;

	/* No copying */
private:
	FileListScript(const FileListScript&);
	FileListScript& operator=(const FileListScript&);
};

#endif        //  #if !defined(__SCRIPTWRITER_H__)

