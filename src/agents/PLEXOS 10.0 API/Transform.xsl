<?xml version="1.0"?> 
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0"
xmlns:rs="urn:schemas-microsoft-com:rowset"
xmlns:z="#RowsetSchema">
<xsl:output method="xml" indent="yes"/>
<xsl:param name="tablename"/>
<xsl:template match="NewDataSet">
<rs:data>
	<xsl:for-each select="./node()[local-name(.)=$tablename]">
		<z:row>
			<xsl:for-each select="@*">
				<xsl:copy-of select="."/>
			</xsl:for-each>
		</z:row>
	</xsl:for-each>
</rs:data>
</xsl:template>
</xsl:stylesheet>
