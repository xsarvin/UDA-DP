
/* ====================================================================
   Copyright 2002-2004   Apache Software Foundation

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
==================================================================== */
        


package org.apache.poi.hssf.record;


import junit.framework.TestCase;

/**
 * Tests the serialization and deserialization of the FrameRecord
 * class works correctly.  Test data taken directly from a real
 * Excel file.
 *

 * @author Glen Stampoultzis (glens at apache.org)
 */
public class TestFrameRecord
        extends TestCase
{
    byte[] data = new byte[] {
        (byte)0x00,(byte)0x00,      // border type
        (byte)0x02,(byte)0x00       // options
    };

    public TestFrameRecord(String name)
    {
        super(name);
    }

    public void testLoad()
            throws Exception
    {

        FrameRecord record = new FrameRecord((short)0x1032, (short)data.length, data);
        assertEquals( FrameRecord.BORDER_TYPE_REGULAR, record.getBorderType());
        assertEquals( 2, record.getOptions());
        assertEquals( false, record.isAutoSize() );
        assertEquals( true, record.isAutoPosition() );


        assertEquals( 8, record.getRecordSize() );

        record.validateSid((short)0x1032);
    }

    public void testStore()
    {
        FrameRecord record = new FrameRecord();
        record.setBorderType( FrameRecord.BORDER_TYPE_REGULAR );
        record.setOptions( (short)2 );
        record.setAutoSize( false );
        record.setAutoPosition( true );


        byte [] recordBytes = record.serialize();
        assertEquals(recordBytes.length - 4, data.length);
        for (int i = 0; i < data.length; i++)
            assertEquals("At offset " + i, data[i], recordBytes[i+4]);
    }
}
