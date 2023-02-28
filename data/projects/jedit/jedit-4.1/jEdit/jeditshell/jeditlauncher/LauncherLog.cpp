/*
 * LauncherLog.cpp - part of jEditLauncher package
 * Copyright (C) 2001 John Gellene
 * jgellene@nyc.rr.com
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
 * $Id: LauncherLog.cpp,v 1.1 2002/01/18 19:31:55 jgellene Exp $
 */

#include "stdafx.h"
#include "LauncherLog.h"


CLtslog* LauncherLog::pLog = 0;
LogArgsFunc LauncherLog::logProc = 0;
HMODULE LauncherLog::hLogModule = 0;

void LauncherLog::Init()
{
	hLogModule = LoadLibrary("ltslog.dll");
	if(hLogModule != 0)
	{
		LogOpenFunc openProc =
			(LogOpenFunc)GetProcAddress(hLogModule, "OpenLog");
		logProc = (LogArgsFunc)GetProcAddress(hLogModule, "LogArgs");
		if(openProc != 0 && logProc != 0)
		{
			pLog = (openProc)("jelaunch.log", Debug);
		}
	}
}

void LauncherLog::Exit()
{
	if(pLog != 0)
	{
		LogCloseFunc closeProc =
			(LogCloseFunc)GetProcAddress(hLogModule, "CloseLog");
		if(closeProc != 0)
			(closeProc)(pLog);
		pLog = 0;
	}
	if(hLogModule)
		FreeLibrary(hLogModule);
}


void LauncherLog::Log(MsgLevel level, const char* lpszFormat, ...)
{
	if(lpszFormat == 0 || pLog == 0) return;
	va_list argList;
	va_start(argList, lpszFormat);
	(logProc)(pLog, true, level, lpszFormat, argList);
	va_end(argList);
}

